> [!IMPORTANT]
> We introduced a new FlyGym 2.x.x API in March 2026, with a complete code rewrite and redesigned interface. This version delivers significantly improved performance:
> 
> - **~10x speed-up** for CPU-based simulations (~2x real-time throughput)
> - **~300x speed-up** for GPU-based simulation via Warp/MJWarp (~60x real-time throughput)
>
> Additional improvements include:
>
> - Improved scene composition workflow
> - Interactive viewer
> - Simplified dependency stack
>
> This version is not backward compatible, and not all features from FlyGym 1.x.x are available. Feature requests can be submitted via Issues on the Github repository. See more information about the changes [here](https://neuromechfly.org/migration/).
>
> Prefer the old version? FlyGym 1.x.x has been migrated to [`flygym-gymnasium`](https://github.com/NeLy-EPFL/flygym-gymnasium). Its documentation has been migrated to [gymnasium.neuromechfly.org](https://gymnasium.neuromechfly.org/).

## Simulating embodied sensorimotor control with NeuroMechFly v2

![](https://github.com/NeLy-EPFL/_media/blob/main/flygym/banner_large.jpg?raw=true)

[**Documentation**](https://neuromechfly.org/) | [**Paper**](https://www.nature.com/articles/s41592-024-02497-y.epdf?sharing_token=jK2FbKWL99-O28WNqrpXWNRgN0jAjWel9jnR3ZoTv0MjiFZczOI3_5wYVxbEbClrTuJzjKyEfhm2kIwso489-ypEsSqlyasWAEsBCvR9WU5poT-q2bblI6hCc7Zji6wb_jZjfXl7KWLbd2pgZTmWvk_ADQ6RuzlnHwvQyipMJzg%3D) | [**Discussion Board**](https://github.com/NeLy-EPFL/flygym/discussions)

![overview_video](https://github.com/NeLy-EPFL/_media/blob/main/flygym/overview_video.gif?raw=true)

This repository contains the source code for FlyGym, the Python library for NeuroMechFly v2, a digital twin of the adult fruit fly *Drosophila* melanogaster that can see, smell, walk over challenging terrain, and interact with the environment (see our [NeuroMechFly v2 paper](https://www.nature.com/articles/s41592-024-02497-y.epdf?sharing_token=jK2FbKWL99-O28WNqrpXWNRgN0jAjWel9jnR3ZoTv0MjiFZczOI3_5wYVxbEbClrTuJzjKyEfhm2kIwso489-ypEsSqlyasWAEsBCvR9WU5poT-q2bblI6hCc7Zji6wb_jZjfXl7KWLbd2pgZTmWvk_ADQ6RuzlnHwvQyipMJzg%3D)).

NeuroMechFly consists of the following components:
- **Biomechanical model:** The biomechanical model is based on a micro-CT scan of a real adult female fly (see our original NeuroMechFly publication). We have adjusted several body segments (in particular in the antennae) to better reflect the biological reality.
- **Vision:** The fly has compound eyes consisting of individual units called ommatidia arranged on a hexagonal lattice. We have simulated the visual inputs on the fly’s retinas.
- **Olfaction:** The fly has odor receptors in the antennae and the maxillary palps. We have simulated the odor inputs experienced by the fly by computing the odor/chemical intensity at these locations.
- **Hierarchical control:** The fly’s Central Nervous System consists of the brain and the Ventral Nerve Cord (VNC), a hierarchy analogous to our brain-spinal cord organization. The user can build a two-part model — one handling brain-level sensory integration and decision making and one handling VNC-level motor control — with an interface between the two consisting of descending (brain-to-VNC) and ascending (VNC-to-brain) representations.
- **Leg adhesion:** Insects have evolved specialized adhesive structures at the tips of the legs that enable locomotion vertical walls and overhanging ceilings. We have simulated these structures in our model. The mechanism by which the fly lifts the legs during locomotion despite adhesive forces is not well understood; to abstract this, adhesion can be turned on/off during leg stance/swing.
- **Mechanosensory feedback:** The user has access to the joint angles, forces, and contact forces experienced by the fly.

This package is developed at the [Neuroengineering Laboratory](https://www.epfl.ch/labs/ramdya-lab/), EPFL.

### Getting Started

For installation, see [the documentation page](https://neuromechfly.org/installation).

To get started, follow [tutorials here](https://neuromechfly.org/tutorials).

---

## Guia de Simulações Locais

Abaixo estão as instruções para instalar e rodar todas as simulações desenvolvidas neste repositório.

### Pré-requisitos

- **macOS** (Apple Silicon) ou Linux
- **Python 3.12** (obrigatório — a versão 3.13+ não é compatível)
- **Homebrew** (macOS)

### 1. Instalação do FlyGym v2 (simulações de caminhada)

```bash
# Instalar uv (gerenciador de pacotes rápido)
brew install uv

# Clonar e instalar
cd ~/Desktop/www/flygym
uv sync --extra dev --extra examples

# Verificar instalação
uv run python -c "import flygym; print('FlyGym OK')"
```

### 2. Instalação do FlyGym v1 (simulações autônomas com olfato)

A versão v1 (`flygym-gymnasium`) possui visão, olfato e controladores neurais (CPG) que foram removidos da v2 em troca de performance.

```bash
mkdir -p ~/Desktop/www/flygym-v1
cd ~/Desktop/www/flygym-v1
python3.12 -m venv .venv
source .venv/bin/activate
pip install "flygym-gymnasium[examples]"

# Se ocorrer erro de libz no macOS:
brew install zlib
ln -sf /opt/homebrew/Cellar/zlib/*/lib/libz.1.dylib /opt/homebrew/lib/libz.1.dylib
```

---

### Simulações Disponíveis

#### Mosca caminhando (vídeo offline) — FlyGym v2

Gera um vídeo de uma mosca caminhando em chão plano usando dados experimentais gravados (replay cinemático).

```bash
cd ~/Desktop/www/flygym
uv run python run_walking.py
# Saída: fly_walking.mp4
```

#### Mosca caminhando (viewer em tempo real) — FlyGym v2

Abre o viewer interativo do MuJoCo com a mosca caminhando ao vivo. No macOS é necessário usar `mjpython`.

```bash
cd ~/Desktop/www/flygym
uv run mjpython run_walking_live.py
# Feche a janela do viewer para parar
```

#### Múltiplas moscas com colisão — FlyGym v2

3 moscas caminhando simultaneamente com detecção de colisão entre elas. Cada uma tem fase de caminhada diferente.

```bash
cd ~/Desktop/www/flygym
uv run mjpython run_multi_flies.py
# Feche a janela do viewer para parar
```

#### Mundo com obstáculos, terreno e comida — FlyGym v2

Ambiente personalizado com rampa, escadas, troncos, pedras, folhas e esferas de comida. 3 moscas caminham no cenário.

```bash
cd ~/Desktop/www/flygym
uv run mjpython run_obstacle_world.py
# Feche a janela do viewer para parar
```

#### Viewer interativo (modelo estático) — FlyGym v2

Abre o viewer MuJoCo com o modelo da mosca para inspeção visual.

```bash
cd ~/Desktop/www/flygym
uv run python scripts/launch_interactive_viewer.py
```

#### Mosca buscando comida com olfato (vídeo) — FlyGym v1

A mosca usa sensores de odor nas antenas para navegar autonomamente até uma fonte de comida (laranja), desviando de fontes aversivas (azul). Usa controlador híbrido CPG + regras reflexivas.

```bash
cd ~/Desktop/www/flygym-v1
.venv/bin/python run_odor_taxis.py
# Saída: outputs/odor_taxis.mp4 e outputs/odor_taxis_trajectory.png
```

#### Mosca buscando comida (visualização em tempo real) — FlyGym v1

Mesma simulação de busca por comida, com visualização em janela OpenCV ao vivo. Pressione `q` para parar.

```bash
cd ~/Desktop/www/flygym-v1
.venv/bin/python run_odor_taxis_live.py
```

#### Foraging sequencial (múltiplas comidas) — FlyGym v1

A mosca navega por 5 fontes de comida espalhadas pelo ambiente, consumindo cada uma que encontra e seguindo para a próxima. Demonstra comportamento de decisão sequencial.

```bash
cd ~/Desktop/www/flygym-v1
.venv/bin/python run_foraging.py
# Saída: outputs/foraging.mp4 e outputs/foraging_trajectory.png
```

#### Aprendizado associativo — FlyGym v1

A mosca aprende qual odor leva à comida verdadeira ao longo de 3 tentativas. Arena com 2 fontes: comida (verde) e isca/decoy (vermelha), cada uma emitindo um odor diferente. Pesos internos são atualizados conforme o resultado de cada tentativa.

```bash
cd ~/Desktop/www/flygym-v1
.venv/bin/python run_learning.py
# Saída: outputs/learning_trial{1,2,3}.mp4 e outputs/learning_comparison.png
```

#### Enxame — competição entre moscas — FlyGym v1

4 moscas com "personalidades" diferentes (Exploradora, Cautelosa, Ousada, Errática) navegam independentemente pelo mesmo ambiente com comida e fontes aversivas. Cada uma tem ganhos olfativos e ruído de decisão distintos. Gera gráfico comparativo com ranking.

```bash
cd ~/Desktop/www/flygym-v1
.venv/bin/python run_swarm_foraging.py
# Saída: outputs/swarm_trajectories.png
```

#### Grooming — limpeza de antenas — FlyGym v1

A mosca caminha, para, levanta as patas dianteiras até as antenas e realiza movimentos de limpeza (rubbing) por 2,5s antes de retomar a caminhada. As poses de grooming foram computadas via cinemática inversa (scipy IK) no modelo MuJoCo. Durante o grooming, o CPG é desativado e as juntas são controladas diretamente.

```bash
cd ~/Desktop/www/flygym-v1
.venv/bin/python run_grooming.py
# Saída: outputs/grooming.mp4
```

---

### Diferenças entre v1 e v2

| Recurso | FlyGym v2 (flygym) | FlyGym v1 (flygym-gymnasium) |
|---|---|---|
| Performance CPU | ~10x mais rápido | 1x (base) |
| Performance GPU | ~300x (Warp/MJWarp) | Não disponível |
| Visão (olhos compostos) | Não disponível | Simulada com ommatidia |
| Olfato (antenas) | Não disponível | Sensores de odor |
| CPG (padrão de caminhada) | Não disponível | Osciladores acoplados |
| Controlador reflexivo | Não disponível | Regras de correção |
| Reinforcement Learning | Não disponível | Interface Gymnasium |
| Múltiplas moscas | Suportado | Suportado |
| Adesão de pernas | Suportado | Suportado |
| Viewer interativo | MuJoCo nativo | MuJoCo nativo |

### Notas para macOS

- `mjpython` é necessário para `mujoco.viewer.launch_passive()` (viewer com controle programático)
- `python` direto funciona para `mujoco.viewer.launch()` (viewer simples)
- Scripts do FlyGym v2 usam `uv run mjpython` ou `uv run python`
- Scripts do FlyGym v1 usam `.venv/bin/python` diretamente