Para baixar o “modelo do cérebro” e colocar no seu projeto, você precisa escolher qual nível quer integrar:

conectoma bruto da FlyWire,
modelo funcional de cérebro inteiro em LIF,
módulo conectômico de visão já acoplável ao FlyGym.

Essas opções não são equivalentes. O paper da Nature de 2024 do cérebro inteiro disponibiliza código e aponta também onde estão os resultados/dados da modelagem. Já o repositório oficial do modelo inclui os arquivos Completeness_783.csv e Connectivity_783.parquet, que são justamente os dados públicos do FlyWire usados para a versão 783.

O caminho mais direto para “baixar o cérebro”

O jeito mais prático hoje é usar o repositório philshiu/Drosophila_brain_model, porque ele já vem preparado para rodar o modelo funcional do cérebro inteiro baseado no conectoma. O README diz que o modelo acompanha o paper, que o código principal está em model.py, e que você pode trocar para a versão pública 783 apontando para Completeness_783.csv e Connectivity_783.parquet.

Use assim no terminal:

git clone https://github.com/philshiu/Drosophila_brain_model.git
cd Drosophila_brain_model
conda env create -f environment.yml
conda activate drosophila_brain_model
jupyter notebook

O passo do conda env create -f environment.yml está no README oficial. O mesmo README diz para usar os notebooks example.ipynb e figures.ipynb como ponto de partida.

Como ativar a versão pública mais nova do conectoma

O próprio README informa que o código original do paper estava configurado para a versão 630, mas que para usar a versão pública 783 você deve alterar o dicionário config para apontar para:

./Completeness_783.csv
./Connectivity_783.parquet

Exemplo:

config = {
    'path_res'  : './results/new',
    'path_comp' : './Completeness_783.csv',
    'path_con'  : './Connectivity_783.parquet',
    'n_proc'    : -1,
}

Esse é, na prática, o “modelo do cérebro” mais próximo do que você está procurando para integrar localmente: um modelo LIF do cérebro inteiro, baseado na conectividade do FlyWire e identidade de neurotransmissores. O artigo descreve esse modelo como um cérebro inteiro de mais de 125 mil neurônios e 50 milhões de sinapses, implementado em Brian2.

Onde estão os arquivos de dados

No repositório philshiu/Drosophila_brain_model, os próprios arquivos aparecem listados na raiz:

Completeness_783.csv
Connectivity_783.parquet

Ou seja, para esse repo você normalmente não precisa sair caçando o dataset manualmente: ele já vem com os arquivos principais necessários para rodar a versão pública 783 do conectoma.

Se você quiser algo mais recente e mais “engenheirado”

Existe também o repo eonsystemspbc/fly-brain, que usa dados do FlyWire v783 e organiza a simulação em múltiplos backends, incluindo Brian2, Brian2CUDA e PyTorch. O README lista os arquivos:

2025_Completeness_783.csv
2025_Connectivity_783.parquet
weight_coo.pkl
weight_csr.pkl

Esse repo também documenta um quickstart com:

conda env create -f environment.yml
conda activate brain-fly
python main.py --brian2-cpu

ou para rodar tudo:

python main.py --t_run 1 --n_run 1 --no_log_file

Isso é útil se você quiser uma base mais moderna para benchmark e experimentação, mas não é o mesmo que “plug and play” com NeuroMechFly.

Se o seu objetivo é usar isso dentro do FlyGym / NeuroMechFly

Aqui tem a parte mais importante: o FlyGym oficial não vem pronto para controlar o corpo inteiro com o modelo completo do cérebro da Shiu et al.. A documentação oficial mostra integração de um modelo conectoma-constrained do sistema visual com o corpo simulado, enquanto a locomoção ainda usa controladores como HybridTurningController / HybridTurningFly.

Então, mesmo baixando o modelo do cérebro inteiro, você não vai simplesmente copiar uma pasta e pronto. Você vai precisar criar uma ponte entre:

saída sensorial do FlyGym,
entrada do modelo neural,
saída neural para neurônios descendentes / comandos motores,
conversão desses comandos em ações do corpo.
Quando usar cada opção

Se o que você quer é só “ter o cérebro baixado” no projeto, vá de philshiu/Drosophila_brain_model. Ele é o caminho oficial mais direto.

Se você quer algo já mais orientado a execução moderna e backends variados, teste eonsystemspbc/fly-brain.

Se você quer algo que já conversa melhor com o FlyGym, o mais fácil hoje é começar pelo modelo visual do flyvis, porque o próprio FlyGym documenta essa integração. O repositório TuragaLab/flyvis é a implementação oficial do modelo conectoma-constrained do sistema visual e oferece documentação e tutoriais próprios.

Em resumo

Para baixar agora e testar no seu projeto:

git clone https://github.com/philshiu/Drosophila_brain_model.git
cd Drosophila_brain_model
conda env create -f environment.yml
conda activate drosophila_brain_model
jupyter notebook

Depois, no código, use a configuração da versão 783:

config = {
    'path_res'  : './results/new',
    'path_comp' : './Completeness_783.csv',
    'path_con'  : './Connectivity_783.parquet',
    'n_proc'    : -1,
}

Isso te dá o modelo funcional do cérebro inteiro. Mas, para ele controlar o corpo do NeuroMechFly/FlyGym, você ainda precisa implementar a camada de integração sensorimotora.