
# MyNanoGPT - Modelo Transformers

![MyNanoGPT](assets/transformers.jpeg)

Baseado no projeto  [nanoGPT](https://github.com/karpathy/nanoGPT) de Andrej karpathy, como parte da disciplina de IA Generativa ministado pelo Prof. [Baldoino Fonseca](https://github.com/baldoinofonseca/baldoinofonseca.github.io) do meu Mestrado do Intituto de Computação da [UFAL](https://ic.ufal.br/pt-br). O MyNanoGPT consta de um repositório simples e rápido em linguagem Python para treinamento/ajuste de GPTs de pequeno e médio porte. No caso do MyNanoGPT, é um fork do nanoGPT usado para estudo do modelo Transformers (rede neural de aprendizado de contexto)

O treinamento é realizado por um código simples e legível: `train.py` que realiza um loop de treinamento padrão. O código foi ajustado para executar em um MacBook com processador M2 com 8 núcleos de CPU e 10 núcleos de GPU. No meu caso não há suporte a CUDA (apenas MPS), porém pode ser configurado caso seu sistema tenha suporte.

## instalação

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` para transformadores huggingface <3 (para carregar pontos de verificação GPT-2)
-  `datasets` para conjuntos de dados huggingface <3 (se você deseja baixar + pré-processar OpenWebText)
-  `tiktoken` para o código BPE rápido da OpenAI <3
-  `wandb` para fazer logging (opcional) <3
-  `tqdm` para barra de progresso <3

## Começando

Para este estudo, vamos usar um dataset que contém o texto das cartas de Eça de Queiroz, um dos maiores escritores da lingua portuguesa [Eça](https://pt.wikipedia.org/wiki/E%C3%A7a_de_Queiroz). O dataset pode ser encontrado em [Dataset](https://www.kaggle.com/datasets/leite0407/ea-de-queiroz), onde além das cartas está disponível as obras em formato txt.

Iniciamos treinando um GPT das cartas de Eça de Queiroz. Primeiro, baixamos ele como um único arquivo (eca.txt->input_eca.txt) na pasta data/ecadequeiuroz_char e o transformamos de texto bruto em um grande fluxo de números inteiros:

```
$ python data/ecadequeiuroz_char/prepare.py
```

Isso vai criar os arquivos `train.bin` e `val.bin` nesse diretório de dados. Agora é hora de treinar seu GPT. O tamanho depende muito dos recursos computacionais do seu sistema:

Agora vamos treinar o GPT com as configurações fornecidas no arquivo de configuração [config/train_eca_char.py](config/train_eca_char.py):

```
$ python train.py config/train_eca_char.py
```

Olhando o arquivo de configuração, veja que estamos treinando um GPT com tamanho de contexto de até 256 caracteres, 384 canais de recursos e é um Transformer de 6 camadas com 6 cabeças em cada camada e 2000 interações de treinamento. Com base na configuração, os pontos de verificação do modelo estão sendo gravados no diretório `--out_dir` `out-eca-char`. Assim que o treinamento terminar, podemos fazer uma amostra do melhor modelo apontando o script de amostragem para este diretório:

```
$ python sample.py --out_dir=out-eca-char
```

Isso gera algumas amostras, por exemplo:

```
Então conservado por brilha do estudito por um farto gente e servil, e tendo o vento e sobre o cavalo, o olho em cima do seu coração. O não mostrou a armada ou que ele, na mulher alma da
vida. E apresentamos a contar que era intacto que ele erguia as restantes dos seus vinte com os que lhe cobria as felicidades com o céu ou o conhecido de uma originação é tão concómoda, que não acompanhava na influência das horas famílias e altas de tremes e das rajagens da Viagem Pública: e o meu Portugal ant
---------------

Para um porto de luta e doce desagrado, um olho, o lenço de um doce de saibras, o riso do seu coração, ao pé da caleche -a fome à espessura da assunta português: e a camisa da Revela que cinco que em dias se acham para que se entraram por cima dos seios da Torre, em os troncos poderos pendiam que se fartaram. A casa do braço, forçando a queixo, por toda a sesta dava Cidade. Mas, desapelou a casa do cavaleiro, ele sempre sacudiu pela ideia da cama de lágrimas envernecidas que a minha casa se e
---------------

– O Cavaleiro Meirinho não sente a mesma Portuguesa e
a dispussão da Rílama que viera alguma reforte medida para cima da cidade, caiu em horas de todo o coração da perdida para se fiscrever o nosso
instante e de bom torpe e modo sobre uma frase, entre ele o anarquista mole da face fortuna que era para tempo instalar do braço amarelo pela escola de Sé. Mas então Napoleães como heróicosos de danos e que o estamos quando todo o querem o copo, hem? Mas, depois de repente! O poesia estava, o café, co
---------------
```

E é isso... 🤩 
