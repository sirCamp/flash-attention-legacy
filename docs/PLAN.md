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
| B7 | Input BF16 o FP32 convertiti in FP16 senza avviso nel percorso kernel: possibile overflow sopra 65504 | modelli BF16 (Gemma, …) | **chiuso in P0.5** |
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

## 1b. Prestazioni misurate (2026-10-21)

Il forward è corretto ma **da circa 6× a 10× più lento di SDPA memory-efficient** di PyTorch, su tutte le architetture (anche dopo le tile Bc=16 di P2.3). I benchmark del README lo mostrano "più veloce" perché confrontano con eager, che materializza la matrice N×N.

| Forward causale, B=2, H=16, N=4096 | nostro (tile iniziali) | nostro (Bc=16, P2.3) | nostro (padding, P5.3) | SDPA |
|---|---|---|---|---|
| V100, d=64 | 22.1 ms | 13.4 ms | **10.1 ms** | 2.34 ms |
| V100, d=128 | 33.0 ms | 29.0 ms | **24.3 ms** | 3.74 ms |
| T4, d=64 | 53.6 ms | 37.4 ms | **26.1 ms** | 4.59 ms |
| T4, d=128 | 64.6 ms | 64.5 ms | **48.8 ms** | 10.4 ms |
| P100, d=64 | 68.1 ms | — | **30.9 ms** | 15.7 ms |
| P100, d=128 | 127 ms | — | **121 ms** | 28.9 ms |
| GTX 1080 Ti, d=64 (H=8) | 151 ms | — | **40.8 ms** | — |

Dopo P5.3 (Nsight Compute, T4, d=64): conflitti di banco in lettura da 389 milioni a **0**, forward da 37.4 a 26.1 ms. Restano occupancy al 12.5% (1 blocco per SM) e stalli su shared memory e barriere: è il limite del disegno "un thread per riga", oggetto di P2.8.

**Backward Pascal** (P5.2, fatto): fwd+bwd con d=128 a 4096 token da 3.1–3.8 s a 0.27–0.43 s (P100 / 1080 Ti), cioè da 9 a 11 volte più veloce.

Nsight Compute sulla T4 (d=64, N=4096):
- tensor core attivi il **2.1%** del picco;
- **92%** delle letture e delle scritture in shared memory in conflitto di banco;
- stallo principale l'attesa della shared memory (44%);
- occupancy 12.5% (1 blocco da 4 warp per SM, 168 registri per thread).

Causa: il disegno "un thread per riga". Softmax, conversione di P e accumulo di O leggono e scrivono la propria riga di S/P/O in shared memory con passo di 32 float, quindi tutti i thread del warp colpiscono lo stesso banco.

Interventi:
1. **P5.3, anticipata**: padding e accessi a 128 bit, una correzione contenuta.
2. **P2.8**: forward alla FA2, con S/P nei registri e i warp che si dividono le righe; è il passo necessario per arrivare al livello di SDPA.

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
| P0.5 | **B7, dtype** (fatto): BF16/FP32 in inferenza girano sul kernel dopo una conversione in FP16 controllata nell'API (`FP16RangeError` se un valore supera 65504 → fallback `fp16_overflow` nel dtype originale). In training (gradienti richiesti) vanno su SDPA (`non_fp16_training`): senza loss scaling i gradienti FP16 andrebbero in underflow. FP32 in inferenza: avviso una tantum che il calcolo è in FP16. In più: il prodotto scalare del forward Pascal non si forma più in FP16 (overflow con prodotti > 65504; anche 2.3–2.9× più veloce sulla 1080 Ti). | S | Fatto: test di overflow, training BF16 e prodotti grandi falliscono sul codice precedente. |
| P0.6 | **Rimandata e unita a P2.2/P2.3** (decisione 2026-10-21: nessuna T4 disponibile, quindi la guardia non sarebbe verificabile). Fino ad allora su Turing il forward d=64/128 solleva un errore esplicito (`cudaFuncSetAttribute`), non dà risultati sbagliati. | — | Vedi P2.2/P2.3. |
| P0.7 | **Casi limite**: `is_causal=None` nei kwargs si interpreta come attributo del modulo, come fa SDPA. Validazione di `cu_seqlens` (monotonia, ultimo valore uguale al totale). Sequenze di lunghezza 0 senza lanci con griglia 0. | S | Test dedicati. |
| P0.8 | **Documentazione dei limiti noti emersi in P0.1–P0.2** (README):<br>• Pascal + `cache_implementation="static"`: HF compila da solo e Triton non supporta sm<7, quindi serve `disable_compile=True`.<br>• Decoder bidirezionale (`is_causal=False`) con cache statica: le chiavi vuote vengono incluse, come fa SDPA di HF.<br>• Doge con cache statica: indicizza la maschera come 4D e solleva un errore. | S | Sezione "Known limitations" aggiornata. |
| P0.8b | **Test riscritti contro eager FP32**: matrice di modelli come test lento opzionale; ogni bug B1–B8 diventa un test di regressione che **fallisce sul codice attuale**. Test parametrizzati con `FLASH_ATTN_LEGACY_FORCE_ARCH`. Training: confronto della loss su 50 step e della 1-cos dei gradienti con SDPA mem-efficient FP16. | M | I test falliscono prima di P0 e passano dopo, su V100, T4 e 1080 Ti. |
| P0.9 | **Nome del backend: deciso di tenere `flash_attn_legacy`** (2026-10-21). Con P0.4 il nome "flash" fa arrivare a noi `cu_seq_lens` dei VLM, quindi un'unica chiamata varlen sul kernel invece del loop Python per immagine. Il monkey-patch di `_check_and_adjust_attn_implementation` resta, e Llama 4 (che solleva per i nomi "flash") arriva con P6.7. | — | Nessuna modifica. |

### P1 — Infrastruttura di benchmark e host varlen — **M**

| ID | Intervento | Dim | Criterio di accettazione |
|---|---|---|---|
| P1.1 | **Fatto (kernel; branch p2-turing)**: `benchmarks/benchmark.py` misura forward e fwd+bwd contro SDPA memory-efficient e math (backend selezionati esplicitamente) ed eager, con media e deviazione standard su 20 ripetizioni (minimo 10) e memoria di picco; `--save` scrive `docs/bench/<gpu>.json`. Baseline registrata su V100, T4, P100 e 1080 Ti (causale, B=2 H=16, N=512–4096, d=64/128). Rapporto legacy/mem-efficient a N=4096: forward V100 4.0×/6.2× (d=64/128), T4 4.9×/4.7×, P100 1.9×/4.6×, 1080 Ti 0.29×/0.37× (SDPA mem-efficient su sm_61 è lentissimo: FP16 a 1/64); fwd+bwd V100 2.8×/4.3×, T4 3.5×/4.5×, P100 0.88×/1.25×, 1080 Ti 0.13×/0.15×. Resta: benchmark a livello di modello (prefill, training), oggi in `tests/test_generation_benchmark.py`. | M | Fatto: baseline su 4 GPU prima di P2.8. |
| P1.2 | **Host varlen**: griglia `(cdiv(max_seqlen_q, Br), num_seqs, H)` senza `block_offsets`, senza `.to(CPU)` e senza la tabella Br hardcoded per SM. Tocca solo il lancio, non la logica del kernel. | S–M | Nessuna sync CPU-GPU nel forward varlen (profiler); test verdi. |

### P2 — Fondamenta dei kernel, Turing e build — **L**

| ID | Intervento | Dim | Criterio di accettazione |
|---|---|---|---|
| P2.1 | **Refactor**: un solo kernel forward e un solo backward con policy `SeqInfo` (batched/varlen) e `Mma` (WMMA sm_70/75, SIMT sm_60/61). Funzione `constexpr` unica per la shared memory e helper comuni. Le feature diventano rami a runtime uniformi, non flag di template, per contenere le istanziazioni. | L | Forward identico bit a bit prima e dopo; nessuna regressione oltre il rumore misurato con l'harness P1.1. |
| P2.2 | **Fatto (branch p2-turing)**: tile scelte a runtime con `with_first_fitting_tile` sulla shared memory per blocco del device (`cudaDevAttrMaxSharedMemoryPerBlockOptin`, con cache; hook di test `FLASH_ATTN_LEGACY_SMEM_LIMIT`). Template di tile con `smem_bytes()` unico; il varlen prende `Br` dalla stessa selezione. | M | T4: 525 test verdi (prima 198 fallimenti); V100 con limite 64 KB: 397 test verdi. |
| P2.3 | **Fatto (branch p2-turing)**: tile forward misurate e unificate a **Bc=16 per ogni head_dim** (`<128,16,d>` per d=32–128, `<64,16,256>`): più veloci delle precedenti su V100 e T4 (forward causale, B=2 H=16 N=4096, V100: d=32 da 13.3 a 8.95 ms, d=64 da 21.0 a 13.4, d=96 da 23.4 a 15.3, d=128 da 33.0 a 29.0; T4: d=32 da 38.3 a 22.7, d=64 da 53.5 a 37.4, d=96 da 55.2 a 43.1) e tutte entro i 64 KB di Turing. Backward con accumulatori dK/dV in frammenti WMMA nei registri (d=256 da 66,5 a ~38,5 KB; 0 spill misurati su sm_70/75). Resta: buffer K/V condiviso, staging del backward su `smem_S`, scrittura `half2`, da valutare con P2.8. | M | Fatto: T4, V100 e 1080 Ti verdi. |
| P2.4 | **Native varlen backward** in un solo lancio, sopra il kernel unificato di P2.1 [R6]. | M | Gradienti come il riferimento; nessuna sync; tempo misurato. |
| P2.5 | **int64 e griglia**: offset a int64, B·H fuori da `gridDim.y`, `#include <mma.h>` fuori dal namespace. | S | Test con B·H > 65535 e, dove la memoria basta, oltre 2^31 elementi. |
| P2.6 | **Build** — *architetture configurabili anticipata (richiesta utente 2026-10-21)*: `TORCH_CUDA_ARCH_LIST` / `FLASH_ATTN_LEGACY_ARCHS` per scegliere le architetture; di default solo quelle supportate dal `nvcc` installato (`nvcc --list-gpu-arch`), con un avviso per quelle escluse (CUDA 13 non compila più sm_60/61/70). Poi: file divisi per (famiglia, head_dim); il build di release compila Pascal solo per sm_60/61, un **build di test** mantiene Pascal anche per sm_70/75 così il percorso forzato resta testabile [R5]. | S–M | CUDA 13 compila per la sola T4 senza errori; con `TORCH_CUDA_ARCH_LIST="7.5"` build in pochi minuti; build di release sotto i 3 minuti; test con Pascal forzato nel build di test. |
| P2.7 | **head_dim 72 e 80** (verificati sulle config). d=80 è già multiplo di 16 (Bc=16); d=72 si porta a 80 con zeri in shared memory per WMMA. Su Pascal servono multipli di 8 per i load vettoriali [R14]. Il costo in tempo di build va misurato. | M | Test unitari per d=72 e d=80; encoder di visione di Qwen2-VL e SigLIP sul kernel. |
| P2.8 | **In corso (branch p2-turing): passi 0, 1, 2 fatti, obiettivo raggiunto su V100 e T4.** Forward causale N=4096 contro SDPA mem-efficient: V100 0.67–1.03×, T4 0.57–0.89× (prima 4–6.6× e 4.7–5.5×). Fatti: corpo a fasce con WMMA e staging per warp, tile a 8 warp con ripiego a 64 KB, copie a 128 bit ovunque e prefetch K/V in registri. Restano: passo 5 (Pascal fp32, P100 ancora 1.7–2.8×), divisione corpo/policy (P2.1) insieme al passo 5, passo 3 (`mma.sync` su Turing) ora opzionale. **Forward riprogettato alla FA2** (unito a P2.1, sostituisce P6.6). S e P restano nei registri dei frammenti (`mma`/WMMA), ogni warp si occupa di una fascia di 16 righe invece di un thread per riga, softmax con riduzioni a livello di warp, O accumulato nei frammenti, niente andata e ritorno in shared memory. Su Pascal lo stesso schema a livello di warp con FMA FP32. Motivo: vedi §1b. Progetto e ordine dei passi (0: preparazione e nuovi test; 1: corpo a fasce + WMMA con staging per warp su sm_70/75; 2: prefetch K/V in registri; 3: `mma.sync` m16n8k8 su Turing; 4: tuning; 5: Pascal fp32 a micro-tile; 6: pulizia) in `docs/analysis/kernels/p2_8_design.md`. | L | Forward a 4096 token entro 1.5× da SDPA memory-efficient su V100 e T4 (prima del redesign 4–6.6×); obiettivo ambizioso concordato: circa 1×. Stessi test di correttezza. |
| P2.9 | **Backward WMMA alla FA2** (Volta/Turing): oggi fwd+bwd a N=4096 è 1.9–2.6× (V100) e 2.4–2.8× (T4) più lento di SDPA mem-efficient, il forward è già alla pari. Stessa impostazione del forward: warp per fasce, dS/dP nei frammenti con staging per warp, dK/dV nei registri, dQ con meno atomiche, prefetch. | L | fwd+bwd entro 1.5× (obiettivo circa 1×) da SDPA mem-efficient su V100 e T4; stessi test. |

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
| P5.1 | Percorso FP32 per sm_61. **In parte anticipato in P0.5**: il prodotto Q·K del forward Pascal è in FP32 (FFMA), 2.3–2.9× più veloce sulla 1080 Ti. Resta: K/V in FP32 in shared memory (meno conversioni), P@V, scelta a runtime per sm_60 vs sm_61, misurata anche su P100 | M | 1080 Ti e P100: forward misurato prima e dopo con l'harness |
| P5.2 | **Fatto (branch p2-turing)**: backward CUDA-core riscritto a tile come FA2: S/P/dP/dS del tile [Br,Bc] calcolati cooperativamente con prodotti a 128 bit su righe padded, P e dS in shared memory con passo Bc+1 (senza conflitti per righe e per colonne), dK/dV in registri per (riga KV, fetta di colonne), dQ sommato su tutto il blocco KV con **una** `atomicAdd` per elemento e blocco (prima una per coppia q/kv: Bc volte di più). 0 spill su sm_60/61 (prima 2–4,7 KB di stack a d≥96). fwd+bwd causale B=2 H=16 N=4096: P100 d=64 788→93 ms, d=128 3094→270 ms; 1080 Ti d=64 802→187 ms, d=128 3830→431 ms. Loader di tile e prodotto scalare a 128 bit ora unici nell'header (prima tre copie). Da misurare in seguito: `atomicAdd` di dQ con le colonne più veloci (meno settori L2 per istruzione) e trip count costante nello step 1. | M | Fatto: test verdi su P100, 1080 Ti, V100/T4 (Pascal forzato). |
| P5.3 | **Fatto (branch p2-turing)**: forward WMMA con righe di S/P padded (+4 float / +8 half) e accessi a 128 bit per riga, softmax che scrive P direttamente in half; forward CUDA-core con Q padded (+8 half), prodotto scalare a 128 bit e P nei registri (niente buffer P). Corpo del blocco condiviso tra kernel batched e varlen (`volta_fwd_block`, `pascal_fwd_block`, anticipo di P2.1). Br=64 provato: differenze del ±10% di segno opposto tra GPU, tenuto Br=128. | S | Fatto: conflitti in lettura a 0 (Nsight, T4); tempi in §1b. |

### P6 — Estensioni (da prioritizzare)

| ID | Intervento | Dim |
|---|---|---|
| P6.1 | Kernel di decode split-KV con GQA | M–L |
| P6.2 | Training BF16/FP32 sul kernel: scaling a potenze di 2 del gradiente dO prima della conversione in FP16 (esatto, il backward è lineare in dO), dQ/dK/dV restituiti in FP32 o nel dtype originale, gestione dell'overflow dei gradienti scalati. Chiude il fallback `non_fp16_training` | M |
| P6.3 | dQ deterministico (kernel dQ separato, vedi P5.2) | M |
| P6.4 | Dropout con Philox | L |
| P6.5 | `torch.compile` e CUDA graph: registrare le operazioni con `torch.library` e togliere le sincronizzazioni host: nel builder della maschera `bool(pad.all())` e `int(q_offset)`; nella funzione di attention `bool(mask.all())` quando il marcatore della maschera 2D si perde (una sync per strato, es. Gemma 3 con cache statica); il controllo di range FP16 per input BF16/FP32 (`amax`, una sync per chiamata nel percorso kernel); la lettura dei segmenti (una sync per forward) [R10] | M |
| P6.6 | ~~`mma.sync` su sm_75~~ → confluito in P2.8 | — |
| P6.7 | **Llama 4: chunked attention** (attention locale a blocchi; transformers solleva un errore per i backend con "flash" nel nome, quindi serve anche un adattamento nel registro HF), dopo P4.1 window nel kernel | M |

---

## 3b. Debito temporaneo (da rimuovere)

Soluzioni provvisorie introdotte durante P0, ciascuna con la voce che la sostituisce. Una voce si chiude solo quando il codice provvisorio è **rimosso**, non solo affiancato.

| Provvisorio | Introdotto in | Sostituito da | Si chiude quando |
|---|---|---|---|
| Fallback eager (FP32, a blocchi di query, senza tensor core, orientato all'inferenza) per `softcap` | P0.3 | P4.2 softcap nel kernel, più P4.1 window (gli strati sliding di Gemma 2 oggi vanno anch'essi su eager) | Gemma 2 sul kernel in tutti gli strati; il motivo `softcap` sparisce dal registro |
| Fallback eager per i sink `s_aux` | P0.3 | P4.3 sink nel kernel, più P4.1 window (gpt-oss alterna strati sliding) | gpt-oss sul kernel in tutti gli strati; il motivo `s_aux` sparisce dal registro |
| Fallback eager per `position_bias` (nessun modello HF 5.x lo passa oggi: copertura difensiva) | P0.3 | da decidere: kernel con bias additivo, oppure errore esplicito | Il motivo `position_bias` sparisce dal registro |
| Training BF16/FP32 → fallback SDPA (`non_fp16_training`) | P0.5 | P6.2 backward con scaling di dO | Nessun motivo `non_fp16_training`; gradienti BF16 entro il rumore anche con loss × 1e-6 |
| `_eager_attention` stessa | P0.3 | — | Si rimuove solo quando nessun motivo del registro ha più l'azione `eager` |
| ~~`cu_seq_lens` solleva `NotImplementedError`~~ | P0.1 | P0.4 | **Chiuso in P0.4**: varlen nel kernel o maschera a segmenti nei fallback; resta `raise` solo per `cu_seq_lens` incoerenti |
| Backward varlen come loop Python per segmento (packing in training, encoder di visione in training) | prima di P0, usato da P0.4 | P2.4 varlen backward nativo | Un solo lancio del kernel, nessuna sincronizzazione host per segmento. Già oggi più veloce del fallback SDPA a blocchi: V100 7.6–15.9 ms contro 24, 1080 Ti 36–267 ms contro 5260, a 8192 token |
| Encoder di visione con head_dim 72/80 (Qwen2-VL, SigLIP) → fallback a segmenti | P0.4 | P2.7 nuovi head_dim | Encoder di visione di Qwen2-VL e SigLIP sul kernel varlen |
| Padding (maschera 2D con zeri o maschera 4D) → fallback SDPA | prima di P0 | P3.1 padding nativo | Batch con padding sul kernel in tutti gli strati |
| Sliding window → fallback SDPA | prima di P0 | P4.1 window nel kernel | Gemma 3, Mistral v0.1 e Phi-3 senza motivo `attention_mask` dovuto alla window |
| Guardia sulla T4 (tile che non entrano → fallback) | P0.6 | P2.2/P2.3 tile a runtime | Nessun fallback per shared memory su T4 |

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

1. **Nome del backend.** *Superata: deciso di tenere il nome "flash" (vedi P0.9).* La review propone di rinominare subito (es. `legacy_attn`), perché finché non arrivano d=72 e d=80 (P2.7) i VLM principali non possono comunque usare il kernel. Il rename porta:
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
