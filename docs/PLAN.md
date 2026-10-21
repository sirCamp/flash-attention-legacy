# flash-attention-legacy — Analisi e piano di lavoro

Stato: **bozza v2, aggiornata dopo la review indipendente**. Nessun punto è ancora implementato.

Fonti: analisi su Tesla V100 (sm_70, CUDA 12.6, torch 2.7.1, transformers 5.20.2):
- [`docs/analysis/hf/report.md`](analysis/hf/report.md) — contratto dell'integrazione HF (sorgente transformers con riferimenti file:line, prove su CPU).
- [`docs/analysis/models/report.md`](analysis/models/report.md) — 10 modelli reali misurati sulla V100, con script di riproduzione.
- [`docs/analysis/kernels/report.md`](analysis/kernels/report.md) — registri, spill, shared memory e occupancy da `ptxas -v` e SASS per sm_60, 61, 70 e 75.
- Review del piano v1: risultati integrati qui sotto, voci marcate con **[R#]**.

Dimensioni: **S** = qualche ora, **M** = 1–3 giorni, **L** = più di 3 giorni.

---

## 1. Sintesi dei risultati

**Il kernel è corretto.**
- Forward: 570 configurazioni, backward: 72, nessun fallimento; errore pari a SDPA FP16.
- 377 test unitari passano.
- Isolando l'attention, i gradienti sono come quelli di SDPA FP16: 1-cos rispetto all'oro FP32 di 0.879/0.887, contro 0.866/0.875.

**Bug silenziosi, tutti nell'integrazione HF.**

| # | Bug | Chi è colpito | Stato |
|---|---|---|---|
| B1 | Prefill con `StaticCache`: il kernel include gli slot vuoti della cache, output degenere (`"is,,,,,,,"`) | tutti con `cache_implementation="static"`; da verificare su cache ibride e sliding (Gemma 2/3), offloaded, decoding assistito, beam search | **confermato su V100** |
| B2 | `softcap` ignorato, sia nel kernel sia nel fallback | Gemma 2 | confermato su V100 |
| B3 | Sink `s_aux` ignorati | gpt-oss | confermato su modello ridotto |
| B4 | Sequenze impacchettate con `use_cache=True` o `cu_seq_lens` trattate come una sola sequenza | training con packing | confermato su V100 |
| B5 | `cu_seq_lens` ignorato **sia nel kernel sia nel fallback SDPA**: l'attention attraversa le immagini | Qwen2-VL-2B (vedi B6), Qwen3-VL e Pixtral (d=64, percorso kernel). Qwen2.5-VL **non** è colpito oggi (tutto su SDPA). | confermato su CPU [R1] |
| B6 | Il controllo dell'head_dim si fa sulla **config** (`hidden_size / num_attention_heads`), non sulla singola chiamata. Sbaglia per i sotto-modelli: in Qwen2-VL-2B calcola 96 invece di 80 e lascia passare il nostro backend. Inoltre sposta interi sotto-modelli su SDPA | VLM; tutti i modelli con head_dim personalizzato | confermato da codice e config [R2] |
| B7 | Input BF16 o FP32 convertiti in FP16 senza avviso nel percorso kernel: possibile overflow sopra 65504 | modelli BF16 (Gemma, …) | da codice [R3] |
| B8 | Su T4 il forward d=64 e d=128 solleva un'eccezione (shared memory oltre i 64 KB per blocco): BERT e Llama non partono | sm_75 | confermato da ptxas, da verificare sulla T4 [R4] |

**Feature mancanti** (oggi su SDPA: corrette, ma senza kernel):
- padding;
- packing;
- sliding window: Gemma 3 22 strati su 26, Gemma 2 13 su 26, Phi-3 e Mistral v0.1 tutti;
- decode (`q_len=1`);
- softcap e sink nel kernel;
- dropout;
- BF16;
- head_dim 72 (SigLIP) e 80 (Qwen2-VL / 2.5-VL);
- `torch.compile` [R10].

**Architetture.**
- **Turing**: vedi B8; anche il backward d=256 fallisce (66,5 KB).
- **Occupancy**: 4–8 warp per SM su 64 ovunque (`my_O[d]` per thread, circa 255 registri).
- **Spill**:
  - fwd d=256 su sm_70: stack di 1768 B;
  - backward CUDA-core a d≥96: `my_dK`/`my_dV` interamente in memoria locale.
- **GTX 1080 (sm_61)**: forward limitato da `HMUL2` FP16, circa 30 volte.
- **Backward Pascal**: un'`atomicAdd` globale per ogni elemento di dQ e per ogni coppia di righe.
- **Build**: circa 8 minuti.

**Prestazioni attuali sulla V100.**
- Prefill a 4096 token: da 1,3 a 1,55 volte più veloce di HF SDPA, con circa metà della memoria. Il confronto va però rifatto contro SDPA mem-efficient con backend selezionato esplicitamente, perché HF SDPA con GQA su sm_70 potrebbe usare il backend "math" e gonfiare il guadagno [R12].

---

## 2. Principi

1. **Nessun risultato sbagliato in silenzio.** Ogni caso non supportato passa a un fallback **che lo applica correttamente** o dà un errore esplicito. La decisione si prende per ogni chiamata, mai sulla config.
2. **Riferimento d'oro: eager FP32.** I criteri si basano sui logit entro il rumore FP16 (eager FP16 contro eager FP32 sullo stesso input), non sull'identità dei token, che si rompe con i quasi-pareggi FP16 [R11].
3. **API compatibile con FA2** (`flash_attn_func`, `flash_attn_varlen_func`, `window_size`, `softcap`, `causal`, `deterministic`). Prendiamo spunto da FA2 e xformers/CUTLASS; se copiamo codice manteniamo la licenza BSD-3.
4. **Validazione su ogni architettura**:
   - V100;
   - percorso Pascal forzato sulla V100, **in un build di test che lo mantiene** e con i test parametrizzati [R5];
   - GTX 1080 Ti;
   - T4.
5. **Omogeneità del codice: ogni cosa si fa in un solo modo, in un solo posto.**
   - **Una sola fonte di verità** per head_dim, tile e shared memory (funzione `constexpr` usata da lancio e controlli), convenzioni di masking, dispatch per architettura e nomi dei parametri, gli stessi dal kernel all'API Python.
   - **Niente copie parallele**: un loader di tile, un set di helper WMMA, un kernel con policy per batched e varlen. Le feature estendono queste strutture.
   - **Stesso schema** per validazione (eccezioni, non `assert`), errori, fallback (registro unico dei motivi) e test (stesso riferimento, stessi helper, stesse tolleranze).
   - **Checklist di review** per ogni PR: esiste già una funzione o un pattern? Il nome segue le convenzioni? Il motivo del fallback è nel registro?
6. **Prima la struttura, poi le feature.** Il lavoro sui kernel (varlen backward, window, softcap, sink) arriva **dopo** il refactor P2.1 [R6].

---

## 3. Fasi

### P0 — Correttezza dell'integrazione (Python, piccole) — **M in totale**

Obiettivo: **nessun output sbagliato** su nessuna architettura. Si correggono solo i casi errati, senza rendere più lenti quelli che oggi sono corretti [R8].

| ID | Intervento | Dim | Criterio di accettazione |
|---|---|---|---|
| P0.1 | **Registro unico dei fallback** e **decisione per singola chiamata**. Si toglie il controllo dell'head_dim dalla config (B6); il fallback si decide in base a dtype, head_dim reale, kwargs (`softcap`, `s_aux`, `cu_seq_lens`, `position_bias`), maschera e se la tile entra nell'architettura. Ogni fallback registra il motivo, con un avviso una tantum. | S | Qwen2-VL-2B: il controllo usa d=80 reale. Test che verifica il motivo registrato per ogni caso. |
| P0.2 | **B1, StaticCache**: quando deleghiamo alla maschera SDPA, la chiamiamo con `allow_is_causal_skip=False`; `None` resta solo nel caso che abbiamo verificato. | S | Logit entro il rumore rispetto a eager FP32 con cache statica, ibrida e sliding (Gemma 2/3), offloaded, decoding assistito e beam search, su TinyLlama, Llama 8B e Gemma 3. |
| P0.3 | **B2, B3, MLA**: con `softcap` o `s_aux` si usa un fallback che li applica (eager con maschera). Con head_dim di V diverso da quello di Q si usa SDPA, che lo supporta [R16]. | S | Gemma 2 e gpt-oss ridotto entro il rumore rispetto a eager FP32. |
| P0.4 | **B4 e B5**: con `cu_seq_lens_q/k` si usa `flash_attn_varlen_func` se l'head_dim è supportato. Altrimenti un fallback che **rispetta i segmenti** (maschera a blocchi diagonali o loop per segmento) [R1]. Il packing ricavato da `position_ids` con `use_cache=False` resta sul fallback SDPA di oggi, che è già corretto, finché P1.3 non lo rende veloce [R8]. | S–M | Batch impacchettato: loss uguale a quella delle sequenze eseguite singolarmente. Encoder di visione di Qwen2-VL-2B, Qwen3-VL e Pixtral (2 e 8 immagini) entro il rumore. Gradiente dei q/k proj (1-cos rispetto all'oro) non peggiore di SDPA FP16 mem-efficient [R7]. |
| P0.5 | **B7, dtype**: BF16 va sul fallback (finché non c'è P5.2). Il downcast da FP32 a FP16 si documenta, con un controllo opzionale di overflow (`amax`). | S | Test con input BF16 vicini al limite FP16: nessun overflow silenzioso. |
| P0.6 | **B8, guardia sulle architetture**: prima del lancio si controlla se la tile entra nella shared memory del device; se non entra, il fallback è registrato (P0.1). È temporaneo: P2.3 lo rende inutile. | S | Sulla T4 BERT e Llama girano (su fallback) e i test non sollevano eccezioni. |
| P0.7 | **Casi limite**: `is_causal=None` nei kwargs si interpreta come attributo del modulo, come fa SDPA. Validazione di `cu_seqlens` (monotonia, ultimo valore uguale al totale). Sequenze di lunghezza 0 senza lanci con griglia 0. | S | Test dedicati. |
| P0.8 | **Test riscritti contro eager FP32**: matrice di modelli come test lento opzionale; ogni bug B1–B8 diventa un test di regressione che **fallisce sul codice attuale**. Test parametrizzati con `FLASH_ATTN_LEGACY_FORCE_ARCH`. Training: confronto della loss su 50 step e della 1-cos dei gradienti con SDPA mem-efficient FP16. | M | I test falliscono prima di P0 e passano dopo, su V100, T4 e 1080 Ti. |
| P0.9 | **Nome del backend**: vedi la decisione 1 in §5. | S | Secondo la scelta. |

### P1 — Infrastruttura di benchmark e host varlen — **M**

| ID | Intervento | Dim | Criterio di accettazione |
|---|---|---|---|
| P1.1 | **Harness di benchmark fisso** [R12]: kernel e modelli, prefill, training e memoria, con media e deviazione standard su almeno 10 ripetizioni. Riferimenti: SDPA mem-efficient e math selezionati esplicitamente, più eager. Risultati versionati in `docs/bench/`. | M | Baseline registrata su V100, T4 e 1080 Ti prima di P2. |
| P1.2 | **Host varlen**: griglia `(cdiv(max_seqlen_q, Br), num_seqs, H)` senza `block_offsets`, senza `.to(CPU)` e senza la tabella Br hardcoded per SM. Tocca solo il lancio, non la logica del kernel. | S–M | Nessuna sync CPU-GPU nel forward varlen (profiler); test verdi. |

### P2 — Fondamenta dei kernel, Turing e build — **L**

| ID | Intervento | Dim | Criterio di accettazione |
|---|---|---|---|
| P2.1 | **Refactor**: un solo kernel forward e un solo backward con policy `SeqInfo` (batched/varlen) e `Mma` (WMMA sm_70/75, SIMT sm_60/61). Funzione `constexpr` unica per la shared memory e helper comuni. Le feature diventano rami a runtime uniformi, non flag di template, per contenere le istanziazioni. | L | Forward identico bit a bit prima e dopo; nessuna regressione oltre il rumore misurato con l'harness P1.1. |
| P2.2 | **Tile scelte a runtime** in base alla shared memory per blocco del device (con cache). L'SM serve solo per tensor core e FP16 lento. Toglie la guardia temporanea di P0.6. | M | Su T4 tutti gli head_dim girano sul kernel. |
| P2.3 | **Tile Turing** (d=64 e d=128: `128×16`; buffer K/V condiviso). Backward d=256 con accumulatori dK/dV in frammenti WMMA. Includere anche le voci dimenticate del report [R15]: tile V100 d=64 a 48 KB (2 blocchi per SM), load a 128 bit, spill del forward d=256 su sm_70. | M | `test_flash_attn.py` completo su T4 e V100; tempi con l'harness. |
| P2.4 | **Native varlen backward** in un solo lancio, sopra il kernel unificato di P2.1 [R6]. | M | Gradienti come il riferimento; nessuna sync; tempo misurato. |
| P2.5 | **int64 e griglia**: offset a int64, B·H fuori da `gridDim.y`, `#include <mma.h>` fuori dal namespace. | S | Test con B·H > 65535 e, dove la memoria basta, oltre 2^31 elementi. |
| P2.6 | **Build**: file divisi per (famiglia, head_dim). Il build di release compila Pascal solo per sm_60/61; un **build di test** mantiene Pascal anche per sm_70/75, così il percorso forzato resta testabile [R5]. | S–M | Build di release sotto i 3 minuti; i test con il Pascal forzato girano nel build di test. |
| P2.7 | **head_dim 72 e 80** (verificati sulle config). d=80 è già multiplo di 16 (Bc=16); d=72 si porta a 80 con zeri in shared memory per WMMA. Su Pascal servono multipli di 8 per i load vettoriali [R14]. Il costo in tempo di build va misurato. | M | Test unitari per d=72 e d=80; encoder di visione di Qwen2-VL e SigLIP sul kernel. |

### P3 — Padding, packing e VLM nativi (dopo P2) — **M–L**

| ID | Intervento | Dim | Criterio di accettazione |
|---|---|---|---|
| P3.1 | **Padding**: da scegliere tra (a) unpad, varlen e repad con gli helper HF, e (b) **key padding nel kernel**: `seqlens_k` e offset di inizio per batch nei parametri, che evita unpad e repad nella generazione con padding a sinistra (dimensione S nel report) [R9]. Proposta: (b) per la generazione, (a) per training e padding a destra; da decidere dopo P2.1. | M | Batch con padding: kernel in tutti gli strati e logit entro il rumore su 3 modelli; gradienti come in P0.4. |
| P3.2 | **Packing nativo** da `position_ids` e `cu_seq_lens` sul varlen backward di P2.4. | S | Loss su 50 step come SDPA; più veloce del fallback (harness). |
| P3.3 | **Encoder di visione** sul kernel: Qwen2-VL/2.5-VL (con le finestre di 64 token circa, che sprecano tile Br=128: valutare Br più piccolo), SigLIP, CLIP, Pixtral, con 1, 3 e 8 immagini. | M | Kernel in tutti gli strati di visione; output entro il rumore. |

### P4 — Feature nei kernel (dopo P2) — **M–L**

| ID | Intervento | Dim | Criterio di accettazione |
|---|---|---|---|
| P4.1 | Sliding window `window_size=(left, right)` nel forward e nel backward, saltando i blocchi | S–M | Gemma 3, Mistral v0.1 e Phi-3 sul kernel oltre la finestra, entro il rumore; tempi contro SDPA mem-efficient (a 8k SDPA può andare in OOM: confronto a 4k più memoria a 8k) [R12] |
| P4.2 | Softcap: `cap·tanh(s/cap)`, con il backward che moltiplica per `(1−t²)` | S–M | Gemma 2 sul kernel, entro il rumore |
| P4.3 | Sink (`row_max=sink`, `row_sum=1`) e riduzione per `d_sink` | S–M | gpt-oss ridotto: top1 come eager FP16 |
| P4.4 | Integrazione: il fallback di P0.3 resta solo per i casi non supportati | S | Matrice dei modelli senza fallback per window, softcap e sink |

### P5 — Prestazioni Pascal (GTX 1080 Ti) — **M–L**

| ID | Intervento | Dim | Criterio di accettazione |
|---|---|---|---|
| P5.1 | Percorso FP32 per sm_61 (K/V in FP32 al caricamento, FFMA). Selezionabile anche con `FORCE_ARCH` nel build di test, per verificarne la correttezza sulla V100 [R5] | M | 1080 Ti: forward misurato prima e dopo con l'harness |
| P5.2 | Backward CUDA-core: fette di colonne di dK/dV e dQ ridotto in shared memory. In alternativa, un kernel dQ separato che risolve insieme le atomic Pascal e il determinismo (P6.3) [R15] | M | 1080 Ti: backward misurato; niente spill a d≤128 |
| P5.3 | Padding delle righe in shared memory contro i bank conflict (tutte le architetture) | S | Profiler e harness |

### P6 — Estensioni (da prioritizzare)

| ID | Intervento | Dim |
|---|---|---|
| P6.1 | Kernel di decode split-KV con GQA | M–L |
| P6.2 | BF16 nativo (conversione in FP16 con saturazione, o FP32 su Pascal) | M |
| P6.3 | dQ deterministico (kernel dQ separato, vedi P5.2) | M |
| P6.4 | Dropout con Philox | L |
| P6.5 | `torch.compile` e CUDA graph: registrare le operazioni con `torch.library` e togliere le sync dal builder della maschera (`bool(pad.all())`) [R10] | M |
| P6.6 | `mma.sync` su sm_75, solo se l'harness lo giustifica | L |

---

## 4. Strategia di test

- **Kernel** (`tests/test_flash_attn.py`): riferimento FP32 e tolleranze attuali; parametrizzati su `FORCE_ARCH` nel build di test.
- **Integrazione HF** (`tests/test_hf_equivalence.py`):
  - riferimento eager FP32, con eager FP16 come rumore di fondo;
  - conteggio delle chiamate al kernel e motivo di ogni fallback;
  - matrice dei modelli come test lento.
- **Training**: loss su 50 step e 1-cos dei gradienti contro SDPA mem-efficient FP16.
- **Benchmark**: harness P1.1, prima e dopo ogni fase.
- **Matrice delle architetture**, per ogni fase:

| | V100 (sm_70) | Pascal forzato su V100 | T4 (sm_75) | GTX 1080 Ti (sm_61) | P100 (sm_60) |
|---|---|---|---|---|---|
| Correttezza | ✓ | ✓ (build di test) | ✓ | ✓ | ✓ manuale (utente) |
| Prestazioni | ✓ | — | ✓ | ✓ | ✓ manuale (utente) |

Il P100 (sm_60, FP16 veloce) è supportato, ma lo testa l'utente manualmente: per ogni fase che tocca Pascal va preparato uno script di test e benchmark da lanciare in un colpo. Le ottimizzazioni Pascal si scelgono a runtime per SM, così non peggiorano sm_60 senza accorgercene. La T4 non è disponibile per ora.

---

## 5. Decisioni

**Approvate dall'utente (2026-10-21)**:
- Tutte le proposte qui sotto: rename ora con alias deprecato, padding come in P3.1, priorità di P6 come elencate, un branch per fase.
- La **P100 è supportata**: test manuale dell'utente sul proprio hardware.
- **Metodo**: una modifica alla volta; ogni modifica è seguita da test (sulle architetture disponibili) e da una code review prima di passare alla successiva.

Testo originale delle proposte:

1. **Nome del backend.** La review propone di rinominare subito (es. `legacy_attn`), perché finché non arrivano d=72 e d=80 (P2.7) i VLM principali non possono comunque usare il kernel. Il rename porta:
   - niente monkey-patch;
   - B5 chiuso gratis, perché i VLM passano al loop per immagine di HF;
   - Llama 4 sbloccato;
   - compatibilità con `fullgraph`.

   Si tornerebbe al percorso `cu_seq_lens` (nome con "flash") con P3.3. L'alternativa è tenere il nome e gestire `cu_seq_lens` già in P0.4. **Proposta: rinominare ora**, tenendo `flash_attn_legacy` come alias deprecato, e rivalutare in P3.3.
2. **Padding**: unpad con gli helper HF (privati: fissare la versione o copiarli, circa 150 righe Apache-2.0), key padding nel kernel, o entrambi (proposta in P3.1).
3. **Hardware**: V100, T4 e GTX 1080 Ti. Serve anche un P100?
4. **Priorità di P6**.
5. **Flusso di lavoro**: un branch per fase con PR e review; le modifiche attuali non committate entrano in un primo commit di base.
6. **Ordine**: Turing (P2.2–P2.3) prima di padding e packing nativi (P3). Il piano già lo prevede. P0.6 mette in sicurezza la T4 subito.
