# AIra — глубокий ресерч литературы (annotated bibliography)

> Назначение: рабочая база знаний программы. Кластеры соответствуют темам из `research/ROADMAP.md`. Для каждой работы — зачем она нам. Ссылки приведены там, где верифицированы поиском; классика указана по названию/выходным данным. Дата сборки: 2026-08.

---

## A. Энергетика мозга и количественная нейронаука (T1)

- **Attwell & Laughlin, “An energy budget for signaling in the grey matter of the brain”, JCBFM, 2001** — базисная оценка АТФ-бюджета: сигналинг ≈ 80% энергии коры, из них постсинаптические механизмы — львиная доля. Вывод для нас: дорога не «умножение», а активация каналов/коммуникация.
- **Harris, Jolivet, Attwell, “Synaptic Energy Use and Supply”, Neuron, 2012** — [sciencedirect.com/science/article/pii/S0896627312007568](https://www.sciencedirect.com/science/article/pii/S0896627312007568) — уточнённый бюджет: ~55% сигнальной энергии приходится на синаптическую передачу; передача ~24 000 АТФ/бит; ~1,64×10⁵ АТФ на везикулу. Калибровка «цены бита».
- **Lennie, “The Cost of Cortical Computation…”, Current Biology, 2003** — [cell.com/current-biology/fulltext/S0960-9822(03)00135-0](https://www.cell.com/current-biology/fulltext/S0960-9822(03)00135-0) — спайк у человека ~2,4×10⁹ АТФ; бюджет поддерживает в среднем лишь ~0,16–0,2 спайка/с/нейрон → объяснение жёсткой разреженности активности. Норма дизайна: средняя частота активности ~единицы Гц макс.
- **Levy & Calvert, “Communication consumes ×35 more energy than computation in the human cortex…”, PNAS, 2021** — коммуникация доминирует над «вычислением»; топология и локальность первичны. Опора для метрики трафика памяти.
- **Sengupta & Stemmler, “Power Consumption During Neuronal Computation”, Proc. IEEE, 2014** — энергетический учёт на уровне каналов/нейронов; order-of-magnitude таблица для симуляторов.
- **Moujahid et al., “Energy demands of diverse spiking cells…”, 2014** — [pubmed.ncbi.nlm.nih.gov/24782749](https://pubmed.ncbi.nlm.nih.gov/24782749) — разные типы нейронов расходуют 7–100 нДж/см² на спайк; «дизайнерские нейроны» имеют разную цену.
- **Karbowski; Sterling & Laughlin “Principles of Neural Design” (MIT Press, 2015)** — систематика «почему мозг так устроен»: разрежение, топология, экономия проводов.
- **Zheng & Meister, “The unbearable slowness of being: Why do we live at 10 bits/s?”, Neuron, 2024** — [cell.com/neuron/fulltext/S0896-6273(24)00808-0](https://www.cell.com/neuron/fulltext/S0896-6273(24)00808-0), препринт [arxiv.org/html/2408.10234v2](https://arxiv.org/html/2408.10234v2) — поведенческий поток ~10 бит/с при 10⁹ бит/с сенсорики; модель «внешнего/внутреннего мозга». Архитектурный ориентир: широкополосный фронтенд + узкое последовательное ядро.

## B. Кредитование вклада без глобального backprop (T2)

**Обзоры:**
- **Mei et al., “A Survey of Backpropagation-free Training For LLMs”, 2024** — [techrxiv](https://www.techrxiv.org/users/756917/articles/728971/master/file/data/main-survey-fwd/main-survey-fwd.pdf?inline=true) — таксономия: FF, PEPITA/MEMPEPITA, LRA, MeZO; оценки памяти/сложности на LLM (PEPITA −30–50% вычислений на длинном контексте; MEMPEPITA −50–94% памяти).
- **“On the Emergence of Forward-Only Algorithms” (обзор хронологии BP-free), 2025** — [techrxiv](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.176739859.93468234/v1?onload=true) — линия BP → CHL → DTP → FA → DNI → DRTP → FF → PEPITA → SigProp → SoftHebb → CCL → TinyFoA; полезна для выбора «ключа поколения».

**Forward-only семейство (обучение только прямыми проходами):**
- **Hinton, “The Forward-Forward Algorithm”, 2022** — добро/зло по локальной цели, два прямых прохода; эталон идеи.
- **Dellaferrera & Kreiman, “Error-driven Input Modulation: PEPITA”, NeurIPS 2022** — ошибка модулирует второй прямой проход; простое и воспроизводимое.
- **Lorberbom et al. / SigProp (“Signal Propagation”)** — аналог обучения по локальному сигналу сверху; вариант «почти без обратного канала».
- **FOTON, “Forward Only Learning for Orthogonal Neural Networks of any Depth”, arXiv 2512.20668** — [arxiv.org/html/2512.20668](https://arxiv.org/html/2512.20668) — ортогональное ограничение чинит «барьер глубины» PEPITA; положительное выравнивание градиентов на десятках слоёв, memory-выгодно. Кандидат в ядро.
- **“Closed-form feedback-free learning with forward projection” (FP), Nat. Commun., 2026** — [nature.com/articles/s41467-026-69161-1](https://nature.com/articles/s41467-026-69161-1) — один прямой проход по датасету, замкнутая регрессия на прямые входы; интерпретируемые мембранные потенциалы; сильный few-shot. Кардинально инженерный подход.
- **Mono-Forward, arXiv 2501.09238** — [arxiv.org/html/2501.09238v1](https://arxiv.org/html/2501.09238v1) — локальные ошибки из одного прямого прохода; на малых задачах дотягивает до BP.
- **NoProp, arXiv 2503.24322 (CoLLAs 2025)** — [arxiv.org/abs/2503.24322](https://arxiv.org/abs/2503.24322) — блоки учатся деноизить таргет независимо (идея диффузии); нет ни прямого, ни обратного сквозного прохода; локальный бэкпроп внутри блока допустим.

**Случайные обратные проекции / целевые распространения:**
- **Lillicrap et al., “Random synaptic feedback weights support error backpropagation” (FA), Nat. Commun., 2016** — случайная фиксированная матрица ошибки работает; базовый «нейромодуляторный» паттерн.
- **Nøkland, “Direct Feedback Alignment”, 2016; Liao et al., “How important is weight symmetry…”, 2016** — вес симметрии не нужен; мост к естественной асимметрии.
- **Lee et al., “Difference Target Propagation”, 2015; Bartunov et al., “Assessing biologically plausible τ” (TP/DTP/KP), NeurIPS 2018** — цели вместо градиентов; честные бенчмарки биоправил.
- **Jaderberg et al., “Decoupled Neural Interfaces (DNI)”, 2017** — синтетические градиенты, снятие time-lock слоёв; идея локальной независимости модулей.
- **Malladi et al., “MeZO”, 2023** — нуль-порядковый файнтюнинг LLM по двум точкам; память как у инференса. Вспомогательный инструмент для больших моделей.

**Энергетические/контрастивные:**
- **Scellier & Bengio, “Equilibrium Propagation”, 2017; Laborieux et al., “Scaling Equilibrium Propagation”, 2021** — двухфазная (свободная/затравленная) локальная динамика ≈ backprop; связь с CIM-аналоговым железом.
- **Movellan (1991) Contrastive Hebbian; GeneRec (O’Reilly 1996)** — исторический фундамент био-правил двух фаз.

## C. Предиктивное кодирование (T3)

- **Rao & Ballard, Nat. Neurosci., 1999** — кортикальная экстраклассика PC (еррор-нейроны/предсказания).
- **Whittington & Bogacz, “An Approximation of the Error Backpropagation…”, Neural Comput., 2017** — PC с локальными Хеббовскими правилами приближает backprop.
- **Song et al., “Can the brain do backpropagation? Exact implementation…”, NeurIPS 2020** — точная эквивалентность на пределе; ключ к «PC = локальный бэкпроп».
- **Millidge, Salvatori, Song, Bogacz, Lukasiewicz, “Predictive Coding: Towards a Future of Deep Learning beyond Backpropagation?”, IJCAI-survey, 2022** — [arxiv.org/abs/2202.09467](https://arxiv.org/abs/2202.09467) — главный обзор: связи с BP, гибкость топологий, генеративность, бенчмарки.
- **Salvatori et al., PC-трансформеры / “PC on graphs”** — применимость к трансформерам и произвольным графам (стреляем в наш спайко-блок).
- **“Error Optimization (EO)” для глубоких PC, 2025+** — снимает экспоненциальное затухание сигнала в глубоких PC-сетях (упомянуто в обзорной выдаче): ремонт главного дефекта масштабирования PC.
- Замеченный в обзорах факт: **Pinchetti et al., “Benchmarking Predictive Coding Networks” (MLJ 2024/2025)** — глубокие PC смеют работать хуже мелких без ремонта; наш чек-лист при построении PC-слоя.

## D. Спайковые сети: модели, кодировки, обучение (T4)

**Обучение:**
- **Neftci, Mostafa, Zenke, “Surrogate Gradient Learning in Spiking Neural Networks”, IEEE Sig. Proc. Mag., 2019** — библия суррогатных градиентов.
- **Wu et al. (STBP), 2018/2019; Zenke & Ganguli, “SuperSpike”, 2018; Shrestha & Orchard (SLAYER), 2018** — классика прямого обучения SNN.
- **Deng et al., “Temporal Efficient Training (TET)”, ICLR 2022** — потеря по моментам времени; снижение числа шагов.
- **Rate-based backprop для SNN, NeurIPS 2024** — [openreview.net/forum?id=wlcm21C4nk](https://openreview.net/forum?id=wlcm21C4nk) — упрощение BPTT через rate-декомпозицию: сопоставимо по точности, дешевле по памяти/вычислениям (CIFAR/ImageNet/DVS).
- **MSG / sparse temporal SNN (arXiv 2406.19645)** — [arxiv.org/html/2406.19645v1](https://arxiv.org/html/2406.19645v1) — прямое обучение с разреженными временны́ми шагами (CIFAR10 95,4% @4 шага; обзорная выдержка конверсий Diet-SNN, Rueckauer, Burst-SNN).
- Конверсии: **Diehl+ 2015 (нормировка), Rueckauer+ 2017 (процентили), Han/Bu (мягкий сброс/бёрсты, квантизация)** — библиография конверсии ANN→SNN для низколатентных сетей.

**Онлайн/трёхфакторные:**
- **Frémaux & Gerstner, “Neuromodulated spike-timing-dependent plasticity… three-factor rules”, Front. Neural Circuits, 2016** — канон обзора третьего фактора.
- **Bellec et al., “A solution to the learning dilemma for recurrent networks of spiking neurons (e-prop)”, Nat. Commun., 2020** — [openreview.net/pdf?id=SkxJ4QKIIS](https://openreview.net/pdf?id=SkxJ4QKIIS) — разложение градиента в локальный eligibility trace × broadcast-сигнал; RTRL изящно; конкурент BPTT на TIMIT.
- **Kaiser, Mostafa, Neftci, “DECOLLE”, 2020** — локальные синтетические градиенты через слойные ридауты; чип-совместимо.
- **Bohnstingl et al., OSTL, 2022/2023** — онлайн-пространственно-временное обучение, близко к BPTT при локальной памяти.
- **“Event-driven eligibility propagation in large sparse networks” (NEST), arXiv 2511.21674** — [arxiv.org/html/2511.21674v1](https://arxiv.org/html/2511.21674v1) — e-prop событийно на миллионах нейронов; задержки, локальность, разряженность — шаблон реализации нашего рантайма.
- **“Three-factor delay learning rules…”** (2026) — обучение синаптических/аксональных задержек поверх e-prop; новый степень свободы для наших слоёв.
- **“NeoHebbian synapses…” Sci. Rep., 2026** — [nature.com/articles/s41598-026-35641-z](https://www.nature.com/articles/s41598-026-35641-z) — ускорение онлайн-обучения на цифровых нейроморфных чипах трёхфакторным правилом.

**Архитектурный поиск и практика:**
- **“Spiking Neural Network Architecture Search: A Survey”, arXiv 2510.14235** — [arxiv.org/html/2510.14235v1](https://arxiv.org/html/2510.14235v1) — пространство NAS для SNN; таблица сильных/слабых сторон SNN (энергоэффективность vs стоимость обучения).
- **Practical tutorial/benchmark MNIST/CIFAR (MDPI AI 2025)** — [mdpi.com/2673-4117/6/11/304](https://www.mdpi.com/2673-4117/6/11/304) — sigma-delta нейроны + direct/rate кодировки: 98,1% MNIST, 83% CIFAR-10 @2 шага; до 3× энерго-выгоды к ANN — рабочие «правила большого пальца».

## E. Спайковые трансформеры и спайковые LM (T9)

- **Spikformer, ICLR 2023 (Zhou et al.)** — первый массовый спайковый ViT-подобный блок (SPS + spiking self-attention).
- **Spike-Driven Transformer (Yao et al., BICLab)** — [github.com/BICLab/Spike-Driven-Transformer](https://github.com/BICLab/Spike-Driven-Transformer) — только маски и сложения во внимании, линейная сложность, 77,1% ImageNet-1K (v2 80,0%), до 87× дешевле ванильного внимания по энергии. Эталон спайкового блока для нас.
- **SpikedAttention, NeurIPS 2024** — [neurips.cc/virtual/2024/poster/94181](https://neurips.cc/virtual/2024/poster/94181) — конверсия Swin/BERT без переобучения (winner-oriented spike shift для softmax): ImageNet 80,0%, BERT на GLUE −0,3 п.п., энергия −42…58%. Доказательство: ANN→SNN мост работает и для трансформеров.
- **SpikingMiniLM, SCIS 2024** — [link.springer.com/article/10.1007/s11432-024-4101-6](https://link.springer.com/article/10.1007/s11432-024-4101-6) — чисто спайковый трансформер для GLUE: 75,5 macro-avg ≈ 98% MiniLMv2, без нормализаций, дистилляция ANN→SNN. Наш эталон «LM на спайках».
- **SpikeLLM (Xing et al., 2024)** — GIF-нейроны поверх LLaMA-архитектуры; первый спайковый путь к LLM-шкале (вместе с Meta-SpikeFormer/Spike-driven V3).
- **WTA-Spikingformer (WE/WD-Spikingformer), arXiv 2604.11321** — [arxiv.org/html/2604.11321v1](https://arxiv.org/html/2604.11321v1) — softmax-free WTA-внимание, masked и causal LM, 16 датасетов; чистый спайк-дизайн без SiLU/softmax/RoPE-артефактов. Сильный кандидат-модуль внимания.
- **SpikeGPT (Zhu et al., 2023)** — спайковая генеративная LM на RWKV-подобной рекуррентности.

## F. Дендритные вычисления и кортикальная микросхема (T5)

- **Guerguiev, Lillicrap, Richards, “Towards deep learning with segregated dendrites”, eLife, 2017** — сегрегация FF/FB входов, локальные ошибки; стартовая модель 2-каморчатого нейрона.
- **Sacramento, Costa, Bengio, Senn, “Dendritic cortical microcircuits approximate the backpropagation algorithm”, NeurIPS 2018** — [arxiv.org/abs/1810.11393](https://arxiv.org/abs/1810.11393) — апикальные дендриты вычисляют ошибку как мисматч латерального предсказания и топ-down сигнала; локально во времени. Наш шаблон компартментного слоя.
- **Richards & Lillicrap, “Dendritic solutions to the credit assignment problem”, Curr. Opin. Neurobiol., 2019; Lillicrap+ “Backpropagation and the brain”, Nat. Rev. Neurosci., 2020** — теоретическая картина.
- **Payeur et al., “Burst-dependent synaptic plasticity can coordinate learning…”, Nat. Neurosci., 2021** — бёрсты как сигнал обучения: дешёвый «флаг события» для нашего правила.
- **Mikulasch et al., локальные ошибки на дендритах (2022–2023)** — «where is the error» нейробиология.
- **Обзор “Dendritic Mechanisms for In Vivo Neural Computations and Behavior”, J. Neurosci., 2022** — [pmc.ncbi.nlm.nih.gov/articles/PMC9665914/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9665914/) — активные дендриты в обучении и поведении; BCI-эксперимент по сомато-дендритной связке.
- **Francioni et al., “Vectorized instructive signals in cortical dendrites”, Nature, 2026** — [nature.com/articles/s41586-026-10190-7](https://www.nature.com/articles/s41586-026-10190-7) — экспериментальное подтверждение: дендриты L5 несут векторные обучающие сигналы (reward/error) по одиночным нейронам; оптогенетический срыв нарушает обучение. Мозг реально делает векторизированное кредитование — опора для T5.

## G. Хеббовское/конкурентное/ассоциативное обучение (T6)

- **Krotov & Hopfield, “Unsupervised learning by competing hidden units”, PNAS, 2019** — HWTA + анти-Хебб, глубокая правдоподобная альтернатива BP для признаков.
- **Moraitis et al., “SoftHebb…”, NCE 2022; Journé et al., “Hebbian Deep Learning Without Feedback”, ICLR 2023** — [github.com/NeuromorphicComputing/SoftHebb](https://github.com/NeuromorphicComputing/SoftHebb) — без обратной связи вообще, Байесовская интерпретация, MNIST 99,4 / CIFAR-10 80,3 / STL-10 76,2 / ImageNet 27,3 (1-5 слоёв). Наш самообучающийся фронтенд.
- **Lagani et al., “FastHebb: scaling hebbian training to ImageNet”, 2022** — соревновательные Хеббовские слои на большой шкале.
- **“Hebbian Learning with Global Direction”, arXiv 2601.21367** — [arxiv.org/html/2601.21367](https://arxiv.org/html/2601.21367) — Хебб + единый глобальный указатель: ResNet-50 на ImageNet в пределах 4% от BP, первый масштаб на ResNet-1202. Компромисс «локально + один модулятор» — гипотеза для нашего дизайна.
- **Modern Hopfield / Ramsauer et al., “Hopfield Networks is All You Need”, 2020; Krotov, “Dense Associative Memory”, 2021/2023** — память↔внимание ↔ энергетические функции; слой-ассоциативная память для нашего стека.

## H. Память, сон, непрерывное обучение (T7)

- **McClelland et al. (CLS theory), 1995; Kumaran, Hassabis, McClelland, TICS 2016** — гиппокамп (быстрая эпизодическая) ⇄ кора (медленная семантическая).
- **Kirkpatrick et al., EWC, PNAS 2017; Zenke et al., Synaptic Intelligence, ICML 2017; al., — базовые анти-забывания для сравнения.
- **“Sleep-like unsupervised replay reduces catastrophic forgetting”, Nat. Commun., 2022** — [nature.com/articles/s41467-022-34938-7](https://www.nature.com/articles/s41467-022-34938-7) — спонтанная активность + локальный Хебб лечит забывание; совместимо с iCaRL. Наш протокол сна.
- **WSCL, “Wake-Sleep Consolidated Learning”, arXiv 2401.08623** — [arxiv.org/abs/2401.08623](https://arxiv.org/abs/2401.08623) — явные фазы wake/NREM/REM, кратко- и долговременная память, «сновидения» для forward transfer; CIFAR/Tiny-ImageNet SOTA в continual.
- **SCM: Sleep-Consolidated Memory для LLM, arXiv 2604.20943** — идеи сна перенесены на семантические графы LLM (ранний сигнал направления).
- **Fusi, Drew, Abbott, “Cascade models of synaptically stored memories”, Neuron 2005; Benna & Fusi 2016** — степенной закон забывания через сложные метапластичные синапсы.
- **Обзор “Theories of synaptic memory consolidation and intelligent plasticity…”, 2024** — [arxiv.org/html/2405.16922v1](https://arxiv.org/html/2405.16922v1) — связь cascade/метапластичности/Laborieux-BNN к непрерывному обучению локально.
- **Zenke & Gerstner, “Hebbian plasticity requires compensatory processes on multiple timescales”, Phil. Trans. R. Soc. B, 2017** — гомеостаз/BCM/метапластичность как обязательные стабилизаторы Хебба. Критический дизайн-принцип.

## I. Эволюция и развитие как источник приоров (T10)

- **Zador, “A critique of pure learning…”, Nat. Commun., 2019** — [nature.com/articles/s41467-019-11786-6](https://www.nature.com/articles/s41467-019-11786-6) — геномное бутылочное горлышко: структура ≫ алгоритм обучения; вывод — оптимизировать топологию/правила, а не только веса.
- **Mordvintsev et al., “Growing Neural Cellular Automata”, 2020** — самоорганизация структуры из локальных правил.
- **Levin lab (морфогенетический интеллект)** — коллективная самоорганизация к целевой морфологии; идеи для «развития» топологии.
- **Miconi et al. (differentiable plasticity), метаобучение правил пластичности**; **“Meta-learning local synaptic plasticity for continual familiarity detection”, 2021** — [biorxiv](https://www.biorxiv.org/content/10.1101/2021.03.21.436287v1.full) — мета-обученная локальная пластичность работает в потоке без фаз.

## J. Эффективные архитектуры последовательностей (T9-мост)

- **Gu & Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces”, 2024; Mamba-2; Mamba-3 (arXiv 2603.15569)** — [arxiv.org/abs/2603.15569](https://arxiv.org/abs/2603.15569) — комплексное состояние, MIMO-форма, state-tracking; O(1) память декодинга. Форма рекуррентного ядра для энергии.
- **Обзоры SSM:** Mamba-360 [arxiv.org/html/2404.16112v1](https://arxiv.org/html/2404.16112v1); “A Survey of Mamba” [arxiv.org/html/2408.01129v8](https://arxiv.org/html/2408.01129v8); “From S4 to Mamba” 2025.
- **Kimi Linear / KDA (Moonshot, 2025)** — поканальный гейт в дельта-правиле; линейное внимание с сильным recall (отдельно проверено ранее в проекте).
- **MoE: обзоры** — [arxiv.org/html/2412.14219v2](https://arxiv.org/html/2412.14219v2) (инференс-оптимизации по всему стеку), “The Rise of Sparse Mixture-of-Experts” [arxiv.org/abs/2602.08019](https://arxiv.org/abs/2602.08019) (централизованная и децентрализованная парадигмы). Связь: MoE ≈ условные «колонки», но требует балансировки и трафика — наш кейс стоимости.
- **Mixture-of-Depths (Raposo et al., 2024)** — условные вычисления по токенам; ручка глубины×энергия.

## K. Нейроморфное железо: состояние и ориентиры (T11)

- **Muir & Sheik, “The road to commercial success for neuromorphic technologies”, Nat. Commun., 2025** — [nature.com/articles/s41467-025-57352-1](https://www.nature.com/articles/s41467-025-57352-1) — карта индустрии 2025: цифровые дизайны вытесняют аналоговые «в первом приближении»; главные барьеры — программная модель и масштабирование.
- **Kudithipudi et al., “Neuromorphic computing at scale”, Nature, 2025** — стандарты, со-дизайн, бенчмарки как условие скачка (обзор есть в [tandfonline](https://www.tandfonline.com/doi/full/10.1080/08956308.2025.2560249)).
- **TrueNorth (Merolla et al., Science 2014); Loihi (Davies et al., IEEE Micro 2018); Loihi 2 (Orchard+ 2021)** — базовые дизайны цифровых нейроморфов.
- **Intel Hala Point (2024)** — [newsroom.intel.com](https://newsroom.intel.com/artificial-intelligence/intel-builds-worlds-largest-neuromorphic-system-to-enable-more-sustainable-ai) — 1152 Loihi 2, 1,15 млрд нейронов, 128 млрд синапсов, 2,6 кВт, 15 TOPS/W INT8 при 10:1 спарсити; до 100× энергии на подходящих задачах. Целевой профиль платформы.
- **IBM NorthPole (Modha et al., Science, 2023)** — без внешней памяти, 25× энергоэффективность против 12-нм GPU на ResNet-50; LLM-результаты: 3B-модель, <1 мс/токен, 72,7× энергия против лучшего GPU — [research.ibm.com/blog/northpole-llm-inference-results](https://research.ibm.com/blog/northpole-llm-inference-results). Урок: сборка памяти в вычисление даёт главную экономию даже без спайков.
- **BrainChip Akida 2.0 / AKD1000 (M.2, ~1 Вт, on-chip incremental learning)** — [edge-ai-vision](https://www.edge-ai-vision.com/2025/01/brainchip-brings-neuromorphic-capabilities-to-m-2-form-factor/) — коммерческий эталон edge-нейроморфа.
- **Innatera SNP; SynSense Speck/Xylo** — sub-мВт always-on сенсорика; рынок подтверждает нишу.
- **SpiNNaker / SpiNNaker2** — миллионы ARM-ядер для биофизически детальных симуляций; мост к нейронауке.
- **Состояние рынка 2025–2026 (пресс-обзоры)** — [nextwavesinsight](https://nextwavesinsight.com/neuromorphic-computing-intel-ibm-enterprise-2026/): разрыв «железо есть — софта/спроса нет»; усиливает нашу ставку на собственный стек.

## L. In-memory / аналоговые вычисления (T11)

- **Обзор “Memristive in-memory computing for AI…” (ResearchGate, 2025)** — [researchgate](https://www.researchgate.net/publication/396837297_In-Memory_Computing_for_AI_Acceleration_and_System_Design) — SRAM/RRAM/PCM компромиссы; MoE на 3D-NVM AIMC снимает параметрическое горлышко.
- **“Review of Memristors for In-Memory Computing and SNN…”, Adv. Intell. Syst., 2025** — [advanced.onlinelibrary.wiley.com](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202500806) — цифры: 38–190 fJ/MAC на кроссбарах; до 56 TOPS/W на отдельных устройствах; PUMA до 2446× энергии на задачу. Эталон целевой энергетики.
- **“In-memory Training on Analog Devices with Limited Conductance States”, arXiv 2510.02516** — [arxiv.org/html/2510.02516v1](https://arxiv.org/html/2510.02516v1) — обучение при грубых состояниях проводимости: алгоритмические обходы (именно наш профиль «алгоритм под железо»).
- **“Current Opinions on Memristor-Accelerated ML Hardware”, arXiv 2501.12644** — [arxiv.org/html/2501.12644v1](https://arxiv.org/html/2501.12644v1) — честный обзор узких мест: ADC/DAC, tiling, обучение on-chip.
- **GWU/NIST: layer ensemble averaging против вариабельности мемристоров (2025)** — [ece.engineering.gwu.edu](https://ece.engineering.gwu.edu/researchers-transform-memristor-flaws-ai-efficiency-gains) — шум железа превращается в ансамбль: приём для noise-aware дизайна.
- **IBM analog AI chip (Ambrogio et al., Nat. Electron., 2023)** — PCM-ядра, near-GPU точность при малой мощности: маяк зрелости AIMC.

## M. Биогибридные вычисления (T12)

- **Kagan et al. (Cortical Labs), DishBrain, Neuron, 2022** — обучение культуры нейронов в игровой среде; протоколы стимуляции/FEP объяснение.
- **Cortical Labs CL1 (2025) и FinalSpark Neuroplatform (2024)** — коммерческий биопроцессор и облачный доступ к органоидам; оценки [privatemarketsnews](https://privatemarketsnews.substack.com/p/biological-computing-the-first-commercial) и [tomshardware](https://www.tomshardware.com/pc-components/cpus/worlds-first-bioprocessor-uses-16-human-brain-organoids-for-a-million-times-less-power-consumption-than-a-digital-chip). Для нас — ориентир энергетики и источник протоколов «обучения живого».
- **Brainoware (Guo et al., Nat. Electron., 2023)** — органоид как физический резервуар для речи.

## N. Резервуарные / физические вычисления (T14)

- **Обзор “Physical reservoir computing — recent advances”, 2019** — [pubmed.ncbi.nlm.nih.gov/30981085](https://pubmed.ncbi.nlm.nih.gov/30981085) — таксономия физических резервуаров.
- **“Neuromorphic overparameterisation and few-shot learning in multilayer physical reservoirs”, Nat. Commun., 2024** — [nature.com/articles/s41467-024-50633-1](https://www.nature.com/articles/s41467-024-50633-1) — ансамбли физических резервуаров, перепараметризация → few-shot; рецепты для нашего P-разведки.
- **ESN vs LSM сравнение (2026)** — [mdpi.com/2673-2688/7/2/70](https://www.mdpi.com/2673-2688/7/2/70) — компромиссы точность/квантование; LSM устойчивее к грубой точности — аргумент за спайковый резервуар.

## O. HDC/VSA (T13)

- **Kleyko et al., “A Survey on Hyperdimensional Computing aka VSA, Part I/II”, ACM CSUR 2022/2023** — [2111.06077](https://ar5iv.labs.arxiv.org/html/2111.06077); [2112.15424]; [hd-computing.com](https://www.hd-computing.com/) — полная карта моделей (MAP, HRR, BSC, SDM), кодирования и применений.
- **Kleyko et al., “Vector Symbolic Architectures as a Computing Framework for Nanoscale Hardware”, Proc. IEEE 2023** — [arxiv.org/abs/2106.05268](https://arxiv.org/abs/2106.05268) — VSA как слой абстракции над стохастическим железом: наш кандидат на «внутренний мозг» (символика) и эпизодическую память.
- **Kanerva, “Sparse Distributed Memory”, 1988; “Hyperdimensional Computing”, 2009** — истоки.

## P. Метрики, бенчмарки, стандарты (T0)

- **Yik et al., “NeuroBench: A Framework for Benchmarking Neuromorphic Computing Algorithms and Systems”, Nat. Commun., 2025** — [nature.com/articles/s41467-025-56739-4](https://www.nature.com/articles/s41467-025-56739-4) — принят нами как стандарт метрик (корректность + сложность модели + системные ватты; сценарии single-stream/real-time/offline).
- **Cheng et al., мультимодальный бенчмарк пяти SNN-фреймворков (SpikingJelly/BrainCog/Sinabs/SNNGrow/Lava), Eng. Appl. AI, 2025** — [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0952197625015453) — SpikingJelly лидер по энергии; BrainCog по сложным задачам. Ориентиры выбора референсов.
- **Open Neuromorphic: SNN library benchmarks** — [open-neuromorphic.org](https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/) — скорость ядер: SpikingJelly-CuPy < SLAYER/EXODUS ×1,5–2 < чистые PyTorch; torch.compile вытягивает Norse к JAX-уровню.
- **Spyx (Heckel et al., 2024)** — [arxiv.org/html/2402.18994v1](https://arxiv.org/html/2402.18994v1) — JIT/JAX для SNN; компромисс гибкость↔скорость для нашего ядра.
- **Sze et al., “Efficient Processing of DNNs: Tutorial and Survey”, Proc. IEEE, 2017 + Horowitz energy numbers (ISSCC 2014)** — таблица стоимостей op/памяти: база нашего энерго-прокси.
- **MLPerf Tiny (2021), SNABSuite** — смежные бенчи для микроконтроллерного профиля.

## Q. Прочие ключевые мостики (общее)

- **Hebb 1949; Bienenstock–Cooper–Munro 1982; von der Malsburg; Willshaw & von der Malsburg (самоорганизация карт)** — теоретический фундамент.
- **Olshausen & Field, “Sparse coding…”, Nature 1996** — разрежённое кодирование как принцип V1.
- **Schultz et al., 1997 (дофамин = TD-error); Joel, Niv, Ruppin 2002 (базальные ганглии = actor-critic)** — карта «нейромодуляторов» для нашего третьего фактора.
- **Eliasmith, “How to Build a Brain” + Nengo** — инженерия больших спайковых моделей (SPA/VSA в спайках) — вдохновение для композиции модулей.
- **Gallego et al., “Event-based Vision: A Survey”, TPAMI 2022** — событийные сенсоры как «сетчатка» входа.

---

## R. Зонная организация и межцентровая коммуникация мозга (цикл 2, вопрос пользователя про зоны/центры)

**Деление на зоны и экономика проводки:**
- **Glasser et al., “A multi-modal parcellation of human cerebral cortex”, Nature 2016** — [nature.com/articles/nature18933](https://www.nature.com/articles/nature18933) — ~180 областей на полушарие: масштаб «зоны» мозга (FF-14).
- **Cherniak, “Component placement optimization in the brain”, J. Neurosci. 1994; глава 2012** — [terpconnect.umd.edu/~cherniak/EvoPrimBrn_12.pdf](https://terpconnect.umd.edu/~cherniak/EvoPrimBrn_12.pdf) — постановка «минимум проводки» и где она работает/нет.
- **Kaiser & Hilgetag, PLOS Comput. Biol. 2006; Human CPO, Network Neuroscience 2023** — [direct.mit.edu/netn/article/7/1/254](https://direct.mit.edu/netn/article/7/1/254/113279/Nonoptimal-component-placement-of-the-human) — коннектом (в т.ч. человека) длиннее минимального; избыток покупает динамику интеграции/сегрегации.
- **Bullmore & Sporns, “The economy of brain network organization”, Nat. Rev. Neurosci. 2012** — [nature.com/articles/nrn3214](https://www.nature.com/articles/nrn3214) — экономический трейд-офф «стоимость проводки ↔ польза топологии»: наша рамка для вариационной задачи о зонах (N7/N8).
- **PLOS Biology 2022, оценка абсолютного числа аксонов белого вещества** — [journals.plos.org](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3001575) — межзонные кортико-кортикальные проекции «очень разрежены»; вся подкорка <1% нейронов коры.
- **Herculano-Houzel et al., “Connectivity-driven white matter scaling and folding in primate cerebral cortex”, PNAS 2010** — [pnas.org/doi/10.1073/pnas.1012590107](https://www.pnas.org/doi/10.1073/pnas.1012590107) — доля нейронов с выходом в белое вещество ~N^{−0.16}: глобальность убывает с ростом мозга (FF-14).

**Скелет и маршрутизация:**
- **van den Heuvel & Sporns, “High-cost, high-capacity backbone for global brain communication”, PNAS 2012** — [pubmed.ncbi.nlm.nih.gov/22711833](https://pubmed.ncbi.nlm.nih.gov/22711833/) — rich club: ~40% стоимости трафика, 69% кратчайших путей, мотив local→feeder→club→feeder→local (FF-16, H-18).
- **Sherman & Usrey, “Transthalamic Pathways for Cortical Function”, J. Neurosci. 2024** — [jneurosci.org/content/44/35/e0909242024](https://www.jneurosci.org/content/44/35/e0909242024) — таламус ВП как гейт межзонной коммуникации: прямые связи не умеют гейтить; основа нашего Т5/H-17.
- **Jones, “The thalamic matrix and thalamocortical synchrony”, TiNS 2001** — [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0166223600019226) — core (точечные каналы) vs matrix (диффузный синхронизирующий слой I) — прототип двухрежимной шины (узкий канал + модулятор).
- **Fries, “A mechanism for cognitive dynamics: neuronal communication through coherence”, TiCS 2005; “Rhythms for Cognition: CTC”, Neuron 2015** — [zuckermaninstitute.columbia.edu](https://zuckermaninstitute.columbia.edu/rhythms-cognition-communication-through-coherence) — окна когерентности = динамическая маршрутизация без перекоммутации; гамма вверх, бета вниз, тета-выборка ~4–8 Гц (F4, FF-17).
- **Bastos et al., “Canonical Microcircuits for Predictive Coding”, Neuron 2012; “Layer and rhythm specificity for predictive routing”, PNAS 2020** — [pnas.org/doi/10.1073/pnas.2014868117](https://www.pnas.org/doi/10.1073/pnas.2014868117) — гамма/поверхностные = остатки вверх, альфа/бета/глубокие = предсказания вниз, предсказанное подавляется (FF-15: межзонный трафик = σ-δ на уровне зон).
- **Murray et al., “A hierarchy of intrinsic timescales across primate cortex”, Nat. Neurosci. 2014** — [nature.com/articles/nn.3862](https://www.nature.com/articles/nn.3862) — собственные времена коры растут вдоль иерархии: якорь гипотезы «иерархия шкал = мультигрид» (H-18).

**Редкий глобальный флаш:**
- **Dehaene & Changeux, “Experimental and theoretical approaches to conscious processing”, Neuron 2011; GNW как сеть вещания, Network Neurosci. 2022** — [direct.mit.edu/netn/article/6/4/1186](https://direct.mit.edu/netn/article/6/4/1186/111960/The-global-neuronal-workspace-as-a-broadcasting) — инвариант «модульность + редкое зажигание» (P3b ~300 мс), бродкаст-дерево с корнем в модуле-источнике (Т6).
- **Cogitate Consortium, “Adversarial testing of GNW and IIT”, Nature 2025 (642, 133–142)** — [doi.org/10.1038/s41586-025-08888-1](https://doi.org/10.1038/s41586-025-08888-1) — зажигание по onset подтверждено частично, по offset почти нет, PFC-декодирование слабее предсказанного: оговорка строгости (§12 F6) — нам нужен инвариант, не феноменология сознания.

**Локальное обучение — теоретическое основание N2:**
- **Millidge, Tschantz, Buckley, “Predictive Coding Approximates Backprop Along Arbitrary Computation Graphs”, Neural Computation 2022** — [arxiv.org/abs/2006.04182](https://arxiv.org/abs/2006.04182) — PC сходится экспоненциально к точным BP-градиентам на произвольных графах локальными правилами (FF-11); база трилеммы FF-12.

---

## S. Воспитание: RLHF / конституции / самокоррекция (узел M7, ALIGNMENT.md)

- **Christiano et al., “Deep RL from Human Preferences”, 2017** — [arxiv.org/abs/1706.03741](https://arxiv.org/abs/1706.03741) — первопринцип: награда из парных людских предпочтений.
- **Ouyang et al., “Training language models to follow instructions with human feedback” (InstructGPT), 2022** — [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155) — 3 стадии SFT→RM→PPO; KL-штраф к опорной политике <<≈ наш каскад-гомеостаз>>.
- **Bai et al., “Constitutional AI: Harmlessness from AI Feedback”, Anthropic 2022** — [arxiv.org/abs/2212.08073](https://arxiv.org/abs/2212.08073) — critique→revision (SL-CAI) + ИИ-судья по конституции (RLAIF); конспект-анализ 2026: «конституция как сжатие интерфейса управления» [vanleke.com](https://vanleke.com/blog/2026-06-25-why-constitutional-ai-can-replace-many-harmlessness-labels.html).
- **Rafailov et al., “Direct Preference Optimization” (DPO), 2023** — [arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290) — лёгкий бейзлайн для ворот M7.
- **Shinn et al., “Reflexion”, 2023** — [arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366) — вербальный RL: фидбек как текст в памяти эпизода <<наш сон-реплей (H-20)>>.
- **Madaan et al., “Self-Refine”, 2023** — [arxiv.org/abs/2303.17651](https://arxiv.org/abs/2303.17651) — критика-ревизия одной моделью; сравнить с producer–critic.
- **Huang et al., “Large Language Models Cannot Self-Correct Reasoning Yet”, ICLR 2024** — [openreview.net/pdf?id=IkmD3fKBPQ](https://openreview.net/pdf?id=IkmD3fKBPQ) — предостережение: без внешнего фидбека самокоррекция вредит → держать внешнего критика и делиберацию-гейт.
- **Kumar et al., “SCoRe: Training self-correction via RL”, DeepMind 2024** — [arxiv.org/abs/2409.12917](https://arxiv.org/abs/2409.12917) — RL на самопорождённых исправлениях.
- **Dhor et al., “Reflexion: LMs that Think Twice” с UTD, 2025** — [openreview.net/forum?id=FDG2G7JDWO](https://openreview.net/forum?id=FDG2G7JDWO) — делиберация по триггеру неуверенности <<наше H-04 дословно>>.
- **Eldan & Li, “TinyStories”, 2023** — [arxiv.org/abs/2305.07759](https://arxiv.org/abs/2305.07759); датасет [huggingface.co/datasets/roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — эталон малого языкового бенча (для M1/M3; в нашей среде HF не виден — генеративный заменитель в `src/aira/corpus.py`).

## Мандатный reading-order (ядро, 15 работ + 3 из цикла 2)

0. (цикл 2) Bastos+ 2020 (зонный протокол); van den Heuvel–Sporns 2012 (скелет); Millidge+ 2022 (PC≈BP точно).
1. Attwell & Laughlin 2001 → 2. Harris/Attwell 2012 (энергетика); 3. Zador 2019 (приоры); 4. Lillicrap 2016 (FA); 5. Bellec 2020 (e-prop); 6. Neftci 2019 (surrogate); 7. Millidge+ 2022 (PC-сurvey); 8. Sacramento+ 2018 (дендриты); 9. Hinton 2022 (FF) + PEPITA 2022; 10. Journé+ 2023 (SoftHebb) + Krotov–Hopfield 2019; 11. Zenke–Gerstner 2017 (стабилизаторы); 12. Sleep replay 2022 + WSCL 2024; 13. Spike-Driven Transformer + SpikingMiniLM + SpikedAttention; 14. NeuroBench 2025 (метрика); 15. Muir–Sheik 2025 + NorthPole 2023/2024 LLM-результаты (куда движется железо).
