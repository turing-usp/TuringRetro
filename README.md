<img src="https://media1.giphy.com/media/70JLXQ1K9ViyKUcBxd/giphy.gif" width="40%" style="float:right"/>

# 🎮 TuringRetro

O **TuringRetro** é o projeto de Aprendizado por Reforço do Turing USP que desenvolve agentes que aprendem sozinhos a jogar jogos retrô!

## 📑 Índice

 - [Como rodar o TuringRetro](#Como-rodar-o-TuringRetro)
   - [🕹️ Rodar um agente treinado](#🕹️-Rodar-um-agente-treinado)
   - [🏋️‍♂️ Treinar um agente](#🏋️‍♂️-Treinar-um-agente)
 - [Sobre o Projeto](#Sobre-o-projeto)
 - [Guia de Instalação](#Guia-de-Instalação)

## Como rodar o TuringRetro

### 🕹️ Rodar um agente treinado

Para rodar o TuringRetro com um agente pré-treinado, deve-se rodar o arquivo _run_rllib.py_, indicando:

 - o nome do jogo: `game`
 - o estado do jogo: `state`
 - o agente salvo: `checkpoint`
 - a quantidade de episódios: `numero_de_episodios`

```bash
python run_rllib.py game state -c checkpoint -e numero_de_episodios
```

Por exemplo, para rodar 1 episódio de um agente de *Super Mario Kart* treinado na pista *Mario Circuit 1*, basta rodar o seguinte comando:

```bash
python run_rllib.py SuperMarioKart-Snes mario1.state -c (inserir checkpoint aqui) -e 1
```

### 🏋️‍♂️ Treinar um agente

Caso deseje treinar um novo agente, deve-se rodar o arquivo _run_rllib.py_, indicando:

 - o nome do jogo: `game`
 - o estado do jogo: `state`
 - a flag de treino: `-t`

```bash
python run_rllib.py game state -t 
```

Por exemplo, para treinar um agente de *Mega Man 2* contra o boss *Airman*, basta rodar o seguinte comando:

```bash
python run_rllib.py MegaMan2-Nes Airman.Normal.Fight.state -t
```

## Sobre o Projeto

TODO.

(Projeto de Aprendizado por Reforço do Grupo Turing utilizando o Gym Retro.)

### Ambientes

#### Mega Man 2

<p align="left">
  <img src="https://media4.giphy.com/media/Nv5PW31wt9jYbHmrt4/giphy.gif" width="200"/>
  <img src="https://media4.giphy.com/media/fIDmLfkDkGiDy2GrP5/giphy.gif" width="200"/>
   <img src="https://media1.giphy.com/media/70JLXQ1K9ViyKUcBxd/giphy.gif" width="200"/> </br>
  <img src="https://media2.giphy.com/media/cJOJBT6iBgPlMTeNuV/giphy.gif" width="200"/>
  <img src="https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/e3831030-47af-4f65-b849-5330fe69ce85/d4xhejk-ccbc3f91-4802-4dc8-a549-f32febbc71a0.png/v1/fill/w_900,h_887,q_75,strp/dr__wily_logo_by_callmemra-d4xhejk.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwic3ViIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsImF1ZCI6WyJ1cm46c2VydmljZTppbWFnZS5vcGVyYXRpb25zIl0sIm9iaiI6W1t7InBhdGgiOiIvZi9lMzgzMTAzMC00N2FmLTRmNjUtYjg0OS01MzMwZmU2OWNlODUvZDR4aGVqay1jY2JjM2Y5MS00ODAyLTRkYzgtYTU0OS1mMzJmZWJiYzcxYTAucG5nIiwid2lkdGgiOiI8PTkwMCIsImhlaWdodCI6Ijw9ODg3In1dXX0.x7B-G0R0Xd0_uOU9kiSalO0tOFnbO_-aI5DOpkPM-LQ" width="200"/>
  <img src="https://media3.giphy.com/media/DWcx4fQNJFT9WuqAsI/giphy.gif" width="200"/>
</p>

#### Super Mario Kart

<img src="https://imgur.com/ZHHFcxJ.gif" width="40%" style="vertical-align:middle"/>

## Guia de Instalação

### Bibliotecas necessárias

-  [RLlib](https://github.com/ray-project/ray) - Biblioteca que utilizamos para usar os algoritmos de RL

-  [Gym-retro](https://github.com/openai/retro) - Biblioteca que utilizamos para emular os jogos e treiná-los

-  [Tensorflow](https://github.com/tensorflow/tensorflow) - Framework para criação e utilização de redes neurais

- você também pode usar o [Pytorch](https://github.com/pytorch/pytorch), mas em nossos testes Tensorflow funciona melhor com o RLlib.

Em seu terminal com Python execute os seguintes comandos para instalar as bibliotecas necessárias:


```bash
pip install tensorflow
pip install ray[rllib]
pip install gym-retro
```

### Instalando jogos já integrados

> Alguns jogos já são integrados com o gym-retro, você pode olhar esta lista [aqui](https://github.com/openai/retro/tree/master/retro/data/stable)

Por questões legais não podemos passar os arquivos dos jogos no repositório, porém recomendamos que você os baixe do projeto [archive](https://archive.org/download/No-Intro-Collection_2016-01-03_Fixed).

Com as ROMs dos jogos baixadas, execute o seguinte comando para passar instalar os jogos no gym-retro:
  
```bash
python3 -m retro.import endereco/do/diretorio/das/ROMs/
```

Para jogos como o Mega Man 2, criamos uma série de estados e cenários de recompensas diferentes dos já instalados no gym-retro, por conta disso será necessário que você baixe o `.zip` do ambiente do jogo desejado e o descompacte no diretório onde o gym-retro está instalado. Você pode localizá-lo com os seguintes comandos em um terminal com Python:

```bash
python
>>> import retro
>>> retro.__file__
```  
Vá então para a pasta `gym-retro/retro/data/stable` e procure a pasta do jogo e a descompacte lá. (caso ele não esteja, vá para seção [Instalando jogos não instalados](###Instalando-jogos-não-integrados))

### Instalando jogos não integrados