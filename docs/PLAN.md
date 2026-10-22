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
| P1.2 | **In parte fatto (branch p2-turing)**: offset dei blocchi Q calcolati sul device (niente loop host né copia sincrona host→device), Br dalla stessa selezione di tile del lancio (niente tabella per SM). Validazione di `cu_seqlens` (P0.7) con una copia device→host, ora opzionale: con `validate=False` (usato dall'integrazione HF, P3.2) nessuna sync, griglia con limite superiore. Piano originale: **Host varlen**: griglia `(cdiv(max_seqlen_q, Br), num_seqs, H)` senza `block_offsets`, senza `.to(CPU)` e senza la tabella Br hardcoded per SM. Tocca solo il lancio, non la logica del kernel. | S–M | Nessuna sync CPU-GPU nel forward varlen (profiler); test verdi. |

### P2 — Fondamenta dei kernel, Turing e build — **L**

| ID | Intervento | Dim | Criterio di accettazione |
|---|---|---|---|
| P2.1 | **Refactor**: un solo kernel forward e un solo backward con policy `SeqInfo` (batched/varlen) e `Mma` (WMMA sm_70/75, SIMT sm_60/61). Funzione `constexpr` unica per la shared memory e helper comuni. Le feature diventano rami a runtime uniformi, non flag di template, per contenere le istanziazioni. | L | Forward identico bit a bit prima e dopo; nessuna regressione oltre il rumore misurato con l'harness P1.1. |
| P2.2 | **Fatto (branch p2-turing)**: tile scelte a runtime con `with_first_fitting_tile` sulla shared memory per blocco del device (`cudaDevAttrMaxSharedMemoryPerBlockOptin`, con cache; hook di test `FLASH_ATTN_LEGACY_SMEM_LIMIT`). Template di tile con `smem_bytes()` unico; il varlen prende `Br` dalla stessa selezione. | M | T4: 525 test verdi (prima 198 fallimenti); V100 con limite 64 KB: 397 test verdi. |
| P2.3 | **Fatto (branch p2-turing)**: tile forward misurate e unificate a **Bc=16 per ogni head_dim** (`<128,16,d>` per d=32–128, `<64,16,256>`): più veloci delle precedenti su V100 e T4 (forward causale, B=2 H=16 N=4096, V100: d=32 da 13.3 a 8.95 ms, d=64 da 21.0 a 13.4, d=96 da 23.4 a 15.3, d=128 da 33.0 a 29.0; T4: d=32 da 38.3 a 22.7, d=64 da 53.5 a 37.4, d=96 da 55.2 a 43.1) e tutte entro i 64 KB di Turing. Backward con accumulatori dK/dV in frammenti WMMA nei registri (d=256 da 66,5 a ~38,5 KB; 0 spill misurati su sm_70/75). Il resto è superato da P2.8 / P2.9. | M | Fatto: T4, V100 e 1080 Ti verdi. |
| P2.4 | **Fatto (branch p2-turing)**: backward varlen in un solo lancio (`backward_varlen`): i due corpi backward prendono `BwdBlockArgs` costruiti da una griglia batched o varlen (ricerca della sequenza in `varlen_seq_of_block`, condivisa col forward), D con il layout di L. La validazione di `cu_seqlens` con sync è opzionale (`validate=False`, P3.2). Piano originale: **Native varlen backward** in un solo lancio, sopra il kernel unificato di P2.1 [R6]. | M | Gradienti come il riferimento; nessuna sync; tempo misurato. |
| P2.5 | **Fatto (branch p2-turing)**: passi e offset a 64 bit (`index_t`) in tutte le struct di parametri e negli argomenti di blocco (le copie di tile tengono un passo di riga `int`: i 64 bit nel ciclo costavano il 30% su Pascal a d=256), `precompute_D` a 64 bit, griglie batched appiattite in `gridDim.x` (niente limite 65535 su B·H), controlli sulle dimensioni nell'host. Test: B·H = 66000, tensori oltre 2^31 elementi batched e varlen (V100 32 GB entrambi; T4 e P100 uno dei due). Piano originale: **int64 e griglia**: offset a int64, B·H fuori da `gridDim.y`, `#include <mma.h>` fuori dal namespace. | S | Test con B·H > 65535 e, dove la memoria basta, oltre 2^31 elementi. |
| P2.6 | **Build** — *architetture configurabili anticipata (richiesta utente 2026-10-21)*: `TORCH_CUDA_ARCH_LIST` / `FLASH_ATTN_LEGACY_ARCHS` per scegliere le architetture; di default solo quelle supportate dal `nvcc` installato (`nvcc --list-gpu-arch`), con un avviso per quelle escluse (CUDA 13 non compila più sm_60/61/70). Poi: file divisi per (famiglia, head_dim); il build di release compila Pascal solo per sm_60/61, un **build di test** mantiene Pascal anche per sm_70/75 così il percorso forzato resta testabile [R5]. | S–M | CUDA 13 compila per la sola T4 senza errori; con `TORCH_CUDA_ARCH_LIST="7.5"` build in pochi minuti; build di release sotto i 3 minuti; test con Pascal forzato nel build di test. |
| P2.7 | **Fatto (branch p2-turing)**: d=72/80 sulle tile da 80 (WMMA) e 96 (CUDA-core, che richiedono d multiplo di 32) con colonne a zero: le copie di tile ricevono le colonne valide (`tile_valid_cols`, costante di compilazione per le tile senza padding), le scritture di O/dQ/dK/dV si limitano alle colonne reali. N=4096 causale, contro SDPA mem-efficient: fwd V100 0.92×, T4 0.71–0.73×, P100 0.94–0.98×; fwd+bwd 0.62–0.74×. Encoder di visione Qwen2-VL (d=80) sul kernel varlen nel test HF; modulo stile SigLIP (d=72). Tempo di build: 4 istanze in più per architettura WMMA (fwd/bwd × batched/varlen), non misurato a parte. Piano originale: **head_dim 72 e 80** (verificati sulle config). d=80 è già multiplo di 16 (Bc=16); d=72 si porta a 80 con zeri in shared memory per WMMA. Su Pascal servono multipli di 8 per i load vettoriali [R14]. Il costo in tempo di build va misurato. | M | Test unitari per d=72 e d=80; encoder di visione di Qwen2-VL e SigLIP sul kernel. |
| P2.8 | **Fatto (branch p2-turing): passi 0, 1, 2, 5, 6 fatti, obiettivo raggiunto su V100 e T4; resta opzionale il passo 3 (`mma.sync` su Turing).** Passo 6: `occupancy.py` rigenera `occupancy_table.md` leggendo i tile dai nomi dei kernel e `kNumWarps` / `smem_bytes()` dagli header (nessuna formula duplicata); spill residui piccoli, il più grande sul forward WMMA d=256 della T4 (98/220 B), possibile seguito. Forward causale N=4096 contro SDPA mem-efficient: V100 0.67–1.03×, T4 0.57–0.89× (prima 4–6.6× e 4.7–5.5×). Fatti: corpo a fasce con WMMA e staging per warp, tile a 8 warp con ripiego a 64 KB, copie a 128 bit ovunque e prefetch K/V in registri. Fatto anche il passo 5: forward Pascal a micro-tile fp32 nel corpo condiviso `flash_attn_fwd_band.h` (P2.1 per il forward: un solo ciclo K/V, mascheramento, softmax, kernel batched/varlen e launcher per tutte le famiglie). P100 N=4096: d=64 27→12.2 ms (0.78× SDPA), d=128 81→52, d=256 1768→285; 1080 Ti d=64 74→15.4, d=128 176→41. Restano: d=128/256 su Pascal (tile TN=1 limitate dalla shared memory, d=256 a un warp per blocco), passo 3 (`mma.sync` su Turing) opzionale. **Forward riprogettato alla FA2** (unito a P2.1, sostituisce P6.6). S e P restano nei registri dei frammenti (`mma`/WMMA), ogni warp si occupa di una fascia di 16 righe invece di un thread per riga, softmax con riduzioni a livello di warp, O accumulato nei frammenti, niente andata e ritorno in shared memory. Su Pascal lo stesso schema a livello di warp con FMA FP32. Motivo: vedi §1b. Progetto e ordine dei passi (0: preparazione e nuovi test; 1: corpo a fasce + WMMA con staging per warp su sm_70/75; 2: prefetch K/V in registri; 3: `mma.sync` m16n8k8 su Turing; 4: tuning; 5: Pascal fp32 a micro-tile; 6: pulizia) in `docs/analysis/kernels/p2_8_design.md`. | L | Forward a 4096 token entro 1.5× da SDPA memory-efficient su V100 e T4 (prima del redesign 4–6.6×); obiettivo ambizioso concordato: circa 1×. Stessi test di correttezza. |
| P2.9 | **Fatto (branch p2-turing)**: backward WMMA alla FA2: ogni warp ha una fascia di 16 chiavi (e una fetta di d per d≥96) e calcola Sᵀ, Pᵀ, dPᵀ, dSᵀ come frammenti, con i passi elemento per elemento sui frammenti (mappa riga/colonna da `wmma_acc_layout`); dK/dV nei registri; dSᵀ condiviso in shared memory per dQ = dS·K su tutti i warp, con atomiche coalescenti. Due barriere per blocco Q (prima 9+). fwd+bwd causale N=4096 contro SDPA mem-efficient: V100 da 1.9–2.6× a 0.65–1.15× (d=64 20.4→11 ms, d=128 47→20.9), T4 da 2.4–2.8× a 0.74–1.23× (d=64 54→20, d=128 124→54). Resta: con la fetta di d (kSplitD>1) i warp della stessa fascia ricalcolano Sᵀ e dPᵀ (+50% di MMA a d=96/128, 2.5× a d=256): dividere la riduzione su d o condividere Pᵀ/dSᵀ; registri alti (220–255, 16 B di spill a d=64 su sm_75); prefetch di Q/dO. | L | Fatto: test verdi su V100, T4, P100, 1080 Ti. |

### P3 — Padding, packing e VLM nativi (dopo P2) — **M–L**

| ID | Intervento | Dim | Criterio di accettazione |
|---|---|---|---|
| P3.1 | **Fatto (branch p2-turing)**: scelta (a) estesa a tutti i casi, ora che il varlen ha forward e backward nativi: il mask builder restituisce la maschera di padding 2D (convenzione FA2), la funzione di attenzione fa unpad sul kernel varlen e repad con `index_copy` (differenziabile: anche il training). Condizione: padding contiguo per riga (altrimenti fallback). Query: causale (solo self-attention) le valide fra gli ultimi q_len slot delle chiavi, con righe di padding a 0 come FA2; bidirezionale (encoder, cross-attention, dove le query non dipendono dalla maschera delle chiavi) tutte le righe, come SDPA. Piano di unpad con una sync per maschera (in cache, anche in inference_mode per le maschere del builder); resta la sync di validazione del varlen per chiamata (P6.5). Nel fallback con padding, le righe senza chiavi visibili sono rese sicure come in HF (NaN su torch < 2.5). Un primo tentativo era sbagliato per la cross-attention con encoder paddato (bloccato in review, test di mutazione aggiunti). Test: Llama e Gemma 3 con padding sinistro/destro entro il rumore, generate con padding sinistro, unpad contro SDPA mascherato con gradienti, GQA e prefill a blocchi. Piano originale: **Padding**: da scegliere tra (a) unpad, varlen e repad con gli helper HF, e (b) **key padding nel kernel**: `seqlens_k` e offset di inizio per batch nei parametri, che evita unpad e repad nella generazione con padding a sinistra (dimensione S nel report) [R9]. Proposta: (b) per la generazione, (a) per training e padding a destra; da decidere dopo P2.1. | M | Batch con padding: kernel in tutti gli strati e logit entro il rumore su 3 modelli; gradienti come in P0.4. |
| P3.2 | **Fatto (branch p2-turing)**: packing e padding in training sul varlen nativo (P2.4). Due correzioni emerse misurando: (1) con autocast fp16 (pesi fp32 + GradScaler, il setup AMP standard) il rotary di HF rende q/k fp32 e tutto il training cadeva nel fallback `non_fp16_training`: sotto autocast fp16 l'attention ora calcola in fp16 come SDPA (cast senza controllo di range, gradienti ammessi); (2) `validate=False` nel varlen (griglia con limite superiore, nessuna sync) per i `cu_seqlens` costruiti dall'integrazione: niente sync per strato. Llama d=128, 8 strati, 3.6k token impacchettati su V100: 214 ms/passo e 8.9 GiB contro 542 ms e 20 GiB del fallback SDPA con maschera a blocchi; a 9k token SDPA va in OOM (noi 16.5 GiB); loss come SDPA (differenza al primo passo 0.002). Piano originale: **Packing nativo** da `position_ids` e `cu_seq_lens` sul varlen backward di P2.4. | S | Loss su 50 step come SDPA; più veloce del fallback (harness). |
| P3.3 | **Fatto (branch p2-turing)**: Qwen2-VL (d=64/80), Qwen2.5-VL (finestre), SigLIP (d=72), CLIP, Pixtral (immagini di dimensioni diverse) con 1, 3 e 8 immagini: tutti gli strati sul kernel (batched o varlen), output entro il rumore di eager FP32. Br più piccolo per le finestre di 64 token di Qwen2.5-VL valutato e scartato: varlen a segmenti da 64 (16k token, H=16, d=80) V100 0.47 ms contro 0.63 di SDPA a batch, T4 1.28 contro 1.15; con tile da 64 righe la T4 scende solo a 1.21 (−6%) e peggiora sui segmenti lunghi. Piano originale: **Encoder di visione** sul kernel: Qwen2-VL/2.5-VL (con le finestre di 64 token circa, che sprecano tile Br=128: valutare Br più piccolo), SigLIP, CLIP, Pixtral, con 1, 3 e 8 immagini. | M | Kernel in tutti gli strati di visione; output entro il rumore. |

### P4 — Feature nei kernel (dopo P2) — **M–L**

| ID | Intervento | Dim | Criterio di accettazione |
|---|---|---|---|
| P4.1 | **Fatto (branch p2-turing)**: `window_size=(left, right)` con la semantica di upstream (righe allineate in basso a destra, −1 illimitato) nel forward e nel backward, batched e varlen; i blocchi fuori finestra sono saltati, quelli interni senza maschera. Integrazione HF: la finestra arriva dal builder della maschera (`local_size` verificato sulla `mask_function`) o dal kwarg `sliding_window`; anche i fallback la applicano. P100, finestra 1024, causale: a 8k forward 11.6 ms contro 121.8 di SDPA mem-efficient, fwd+bwd 71 contro 844, meno memoria. Riordino delle FMA QK su Pascal (conflitti di banco): d=128 forward causale da 51.8 a 34 ms. Piano originale: Sliding window `window_size=(left, right)` nel forward e nel backward, saltando i blocchi | S–M | Gemma 3, Mistral v0.1 e Phi-3 sul kernel oltre la finestra, entro il rumore; tempi contro SDPA mem-efficient (a 8k SDPA può andare in OOM: confronto a 4k più memoria a 8k) [R12] |
| P4.2 | **Fatto (branch p2-turing)**: `softcap` (API di upstream) nel forward e nel backward di tutte le famiglie, batched e varlen. Una sola mappa dei punteggi (`ScoreMap`): x = c·tanh(s·scale/c) nel dominio log2, dx/ds = scale·(1−t²); il ramo è uniforme e fuori dai cicli per elemento (con il ramo dentro, il backward WMMA perdeva fino al 14% anche senza softcap). Costo del softcap: P100 +3%, V100 +8–15%. Contro eager FP16 con softcap: V100 0.13–0.68×, P100 d=128 0.40–0.72×, memoria 30–150× inferiore; **P100 d=256 2.4× più lento di eager** (il limite noto di Pascal d=256, che è la head_dim di Gemma 2 2B/9B). Integrazione: Gemma 2 sul kernel in tutti gli strati (sliding con la finestra); il decode con softcap va su eager, mai su SDPA. Piano originale: Softcap: `cap·tanh(s/cap)`, con il backward che moltiplica per `(1−t²)` | S–M | Gemma 2 sul kernel, entro il rumore |
| P4.3 | **Fatto (branch p2-turing)**: `sinks` (un logit per testa di Q, kwarg HF `s_aux`) nel forward e nel backward di tutte le famiglie, batched e varlen. Il softmax online parte dallo stato di una chiave con logit = sink e valore 0 (`m = sink`, `l = 1`), quindi L include il sink e dQ/dK/dV non cambiano; d_sink[h] = −Σ exp(sink − L)·D, ridotto in `precompute_D` (un atomic per warp). −inf = nessun sink. Costo: V100 fwd invariato, fwd+bwd +0–3%; contro il fallback eager: V100 0.05–0.32×, memoria a 4k 32 MB contro 6 GB. Integrazione: gpt-oss sul kernel in tutti gli strati (sliding con la finestra); il decode con sink va su eager, mai su SDPA. Piano originale: Sink (`row_max=sink`, `row_sum=1`) e riduzione per `d_sink` | S–M | gpt-oss ridotto: top1 come eager FP16 |
| P4.4 | **Fatto (branch p2-turing)**: window, softcap e sink sul kernel in tutti i modelli della matrice (Llama, Mistral, Phi-3, Gemma 2, Gemma 3, gpt-oss: prefill senza fallback, un solo test `test_model_matrix_on_kernel`). `position_bias` non è più difensivo: in transformers 5.20 la famiglia T5 (T5, mT5, LongT5, UDOP, Switch Transformers, ...) lo passa all'attention interface, e tutti i suoi strati andavano su eager FP32; ora va su SDPA con il bias nella maschera (come l'SDPA di HF), differenziabile nel bias. Eager resta solo per softcap e sink fuori dal kernel (decode, fino a P6.1). Piano originale: Integrazione: il fallback di P0.3 resta solo per i casi non supportati | S | Matrice dei modelli senza fallback per window, softcap e sink |

### P5 — Prestazioni Pascal (GTX 1080 Ti) — **M–L**

| ID | Intervento | Dim | Criterio di accettazione |
|---|---|---|---|
| P5.1 | **Forward fatto (P2.8 passo 5)**: Q, K, V in FP32 in shared memory (convertiti una volta) e P@V in FP32 su sm_60 e sm_61. Restano aperti, da discutere: il backward CUDA-core legge ancora half dalla shared memory e converte nei prodotti (già più veloce di SDPA mem-efficient a d=128: P100 fwd+bwd 150 contro 214 ms); FP16x2 (HFMA2, doppia velocità) solo su sm_60, con il problema della precisione degli accumuli. Piano originale: Percorso FP32 per sm_61. **In parte anticipato in P0.5**: il prodotto Q·K del forward Pascal è in FP32 (FFMA), 2.3–2.9× più veloce sulla 1080 Ti. Resta: K/V in FP32 in shared memory (meno conversioni), P@V, scelta a runtime per sm_60 vs sm_61, misurata anche su P100 | M | 1080 Ti e P100: forward misurato prima e dopo con l'harness |
| P5.2 | **Fatto (branch p2-turing)**: backward CUDA-core riscritto a tile come FA2: S/P/dP/dS del tile [Br,Bc] calcolati cooperativamente con prodotti a 128 bit su righe padded, P e dS in shared memory con passo Bc+1 (senza conflitti per righe e per colonne), dK/dV in registri per (riga KV, fetta di colonne), dQ sommato su tutto il blocco KV con **una** `atomicAdd` per elemento e blocco (prima una per coppia q/kv: Bc volte di più). 0 spill su sm_60/61 (prima 2–4,7 KB di stack a d≥96). fwd+bwd causale B=2 H=16 N=4096: P100 d=64 788→93 ms, d=128 3094→270 ms; 1080 Ti d=64 802→187 ms, d=128 3830→431 ms. Loader di tile e prodotto scalare a 128 bit ora unici nell'header (prima tre copie). Da misurare in seguito: `atomicAdd` di dQ con le colonne più veloci (meno settori L2 per istruzione) e trip count costante nello step 1. | M | Fatto: test verdi su P100, 1080 Ti, V100/T4 (Pascal forzato). |
| P5.3 | **Fatto (branch p2-turing)**: forward WMMA con righe di S/P padded (+4 float / +8 half) e accessi a 128 bit per riga, softmax che scrive P direttamente in half; forward CUDA-core con Q padded (+8 half), prodotto scalare a 128 bit e P nei registri (niente buffer P). Corpo del blocco condiviso tra kernel batched e varlen (`volta_fwd_block`, `pascal_fwd_block`, anticipo di P2.1). Br=64 provato: differenze del ±10% di segno opposto tra GPU, tenuto Br=128. | S | Fatto: conflitti in lettura a 0 (Nsight, T4); tempi in §1b. |
| P5.4 | **Fatto (branch p2-turing)**: Pascal d=256 (Gemma 2/3): la shared memory fp32 lasciava un solo blocco di 16 righe, cioè **un warp per SM** sulla P100 (metà delle unità FMA ferme, latenze scoperte). Ora i warp di una banda si dividono le colonne (`kSplit` nel tile): S parziale per fetta di colonne, somma in shared memory con una barriera nominata per banda, stesso softmax, O per fetta (16 registri invece di 128). S parziale in shared memory per lane (senza conflitti di banco). Causale B=2 H=16 N=4096: P100 288 → 99 ms (kSplit 2/4/8 = 153/130/120 prima della correzione dei conflitti), 1080 Ti 196 → 121; SDPA mem-efficient 58 / 1049, eager 110 / 141. Il forward da solo resta sopra SDPA (P100 99 contro 58 ms) per il limite di Bc=8 (poco lavoro per blocco K/V): prossimi passi possibili, più invasivi, K e V non residenti insieme (Bc=16) oppure Q e K a fette di colonne come i kernel cutlass. Backward d=256: stesso limite (36 KB, un blocco per SM sulla P100), tile da 8 warp invece di 4: fwd+bwd P100 706 → 398 ms (SDPA mem-efficient 443, math 271), 1080 Ti 664 → 566 (math 353). | M | P100 d=256 almeno pari a SDPA mem-efficient |

### P7 — Backward sui tensor core e Turing `mma.sync` (ordine scelto il 2026-10-22: questo, poi P6.1, P6.2, P6.5)

| ID | Intervento | Dim | Criterio di accettazione |
|---|---|---|---|
| P7.1 | **Fatto**: `precompute_D` coalescente (8 lane per riga): V100 0.76 → 0.09 ms, T4 2.62 → 0.28 | S | — |
| P7.2 | **In corso**: backward Turing con `mma.sync` m16n8k8 e `ldmatrix` (`BwdMma`, progetto e misure in `docs/analysis/kernels/bwd_design.md`): P^T e dS^T restano nei registri come operandi A, niente buffer per warp, d=128 a 8 warp sulla T4. Backward T4 d=128 53.7 → 32.3 ms, d=64 18.2 → 16.6; poi S^T / dP^T divisi per query tra i warp di una banda (P^T / dS^T scambiati in shared memory con una barriera di banda): d=128 26.9, d=256 168 → 136, d=64 14.6 (kSplitD 2). fwd+bwd T4 N=4096 contro SDPA mem-efficient: d=64 0.86×, d=128 0.83×, d=256 0.91×. Famiglia scelta dall'arch del codice che gira (`compiled_arch_for`); sm_8x usa `BwdMma` tramite il PTX 7.5 (non testato). | L | fwd+bwd T4 entro ~1× SDPA mem-efficient |
| P7.3 | **Fatto**: forward Turing con `mma.sync` (`TuringFwd`, `flash_attn_fwd_turing.cu`; P2.8 passo 3): S e O negli accumulatori m16n8, Q nei registri fino a d=128, softmax online sugli accumulatori (`BandSoftmax<2,4>`), P come operando A di P V, nessuno staging. Forward T4 N=4096 causale: d=32 2.74 → 1.19 ms, d=64 3.64 → 2.14, d=96 6.14 → 2.95, d=128 7.12 → 4.13, d=256 18.4 → 16.4. Il WMMA resta per sm_70 (e FORCE_ARCH=volta). | M | — |
| P7.4 | **In corso**: V100, WMMA con lo stesso split per query di S^T / dP^T tra i warp di una banda (P^T nel buffer della banda, dS^T dal buffer condiviso, barriera di banda). Backward kernel V100 N=4096: d=96 16.6 → 12.5 ms, d=128 19.3 → 14.6, d=64 12.0 → 11.3, d=32 5.6 → 5.2, d=256 50.8 → 49.4; fwd+bwd contro SDPA mem-efficient d=128 1.30× → 1.05×, d=256 1.02×; d=64 con `<64,64,64,2>` (8 warp, split per query) 13.7 → 10.3 ms, 0.96×. Alternativa `mma.m8n8k4` se serve. | M | fwd+bwd V100 entro ~1× SDPA |

### P6 — Estensioni (da prioritizzare)

| ID | Intervento | Dim |
|---|---|---|
| P6.1 | **In corso** (progetto `docs/analysis/kernels/decode_design.md`, passi a–g): `flash_attn_with_kvcache` con l'ordine posizionale di upstream, un kernel CUDA-core per tutte le architetture (decode limitato dalla banda), cache fp16/bf16 letta senza copie (anche il layout HF come vista trasposta). Passo a fatto: cache contigua, R ≤ 4 righe per testa KV, causal / window / softcap / sinks, lse. Passo b fatto: split delle chiavi (automatico senza letture dal device, o forzato fino a 128) e combinazione con lo stesso `decode_merge`. T4 decode GQA 32/8 d=128: B=1 contesto 16k 3.27 → 0.32 ms (SDPA 3.86), B=8 4k 0.58 ms (SDPA 6.05), ~200 GB/s su 320. Passo c fatto: fino a 8 righe per testa KV (q in shared memory interleaved; gpt-oss 64/8, Qwen2 28/4); T4 64/8 d=64: 0.06–2.8 ms contro 0.19–30 ms di SDPA, ma ~60–100 GB/s: limitato dalle istruzioni per riga (passo g). Passo d fatto: `cache_leftpad`. Passo f fatto: i passi di decode di HF sul kernel (cache dinamiche, statiche e offloaded, padding a sinistra dalla maschera 2D o dalla 4D `[B,1,1,kv]` delle cache statiche, finestra, softcap, sink), una sincronizzazione host per passo con padding; restano su SDPA FP32, più di 8 righe per testa KV, gradienti attivi. Restano: e (cache paginata) e g (tuning: righe > 4, split dimensionati sulla capacità). | M–L |
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
| ~~Fallback eager (FP32, a blocchi di query, senza tensor core, orientato all'inferenza) per `softcap`~~ **risolto da P4.2** (resta solo per il decode con softcap) | P0.3 | P4.2 softcap nel kernel, più P4.1 window (gli strati sliding di Gemma 2 oggi vanno anch'essi su eager) | Gemma 2 sul kernel in tutti gli strati; il motivo `softcap` sparisce dal registro |
| ~~Fallback eager per i sink `s_aux`~~ **risolto da P4.3** (resta solo per il decode con sink) | P0.3 | P4.3 sink nel kernel, più P4.1 window (gpt-oss alterna strati sliding) | gpt-oss sul kernel in tutti gli strati; il motivo `s_aux` sparisce dal registro |
| ~~Fallback eager per `position_bias`~~ **risolto da P4.4**: SDPA con il bias nella maschera (la famiglia T5 lo usa in tutti gli strati; il kernel con bias additivo e il suo gradiente resta un'estensione possibile, P6) | P0.3 | P4.4 | Nessun motivo del registro con azione `eager` |
| Training BF16/FP32 → fallback SDPA (`non_fp16_training`) | P0.5 | P6.2 backward con scaling di dO | Nessun motivo `non_fp16_training`; gradienti BF16 entro il rumore anche con loss × 1e-6 |
| `_eager_attention` stessa: nessun motivo del registro ha più l'azione `eager`; dopo P6.1 (decode sul kernel) resta solo per softcap e sink nei fallback veri (FP32, head_dim non supportati, maschere con buchi) | P0.3 | P6.1 decode con softcap e sink (fatto) | Da decidere: rimuoverlo richiede softcap/sink anche in quei casi |
| ~~`cu_seq_lens` solleva `NotImplementedError`~~ | P0.1 | P0.4 | **Chiuso in P0.4**: varlen nel kernel o maschera a segmenti nei fallback; resta `raise` solo per `cu_seq_lens` incoerenti |
| ~~Backward varlen come loop Python per segmento~~ **risolto da P2.4** (un lancio) | prima di P0, usato da P0.4 | P2.4 varlen backward nativo | Un solo lancio del kernel, nessuna sincronizzazione host per segmento. Già oggi più veloce del fallback SDPA a blocchi: V100 7.6–15.9 ms contro 24, 1080 Ti 36–267 ms contro 5260, a 8192 token |
| ~~Encoder di visione con head_dim 72/80 (Qwen2-VL, SigLIP) → fallback a segmenti~~ **risolto da P2.7 e P3.3** | P0.4 | P2.7 nuovi head_dim | Encoder di visione di Qwen2-VL e SigLIP sul kernel varlen |
| ~~Padding (maschera 2D con zeri o maschera 4D) → fallback SDPA~~ **risolto da P3.1** (resta il fallback per buchi dentro una riga) | prima di P0 | P3.1 padding nativo | Batch con padding sul kernel in tutti gli strati |
| ~~Sliding window → fallback SDPA~~ **risolto da P4.1** | prima di P0 | P4.1 window nel kernel | Gemma 3, Mistral v0.1 e Phi-3 senza motivo `attention_mask` dovuto alla window |
| ~~Ciclo sui blocchi di query del backward duplicato tra `BwdWmma` e `BwdMma`~~ **risolto**: `bwd_q_loop` e `bwd_p_ds` condivisi dai tre backward, `band_barrier` unica in `flash_attn_common.h`. Era: ciclo duplicato tra `BwdWmma` e `BwdMma` (caricamenti di Q/dO/L/D, maschera e ScoreMap per elemento, passo dQ), e `band_barrier` ripetuta nel forward Pascal e nel backward Turing | P7.2 | Un ciclo condiviso come `fwd_band_loop` (corpo per famiglia), una sola barriera di banda in `flash_attn_common.h` | Una sola copia del ciclo e della barriera; test verdi su tutte le GPU |
| ~~Guardia sulla T4 (tile che non entrano → fallback)~~ **risolto da P2.2/P2.3**: tile scelti a runtime con `with_first_fitting_tile`, nessun motivo di fallback per la shared memory | P0.6 | P2.2/P2.3 tile a runtime | Nessun fallback per shared memory su T4 |

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
