### <コードサンプル>
[PDF版 AlphaZero 深層学習・強化学習・探索 人工知能プログラミング実践入門 | ボーンデジタル (borndigital.co.jp)](https://www.borndigital.co.jp/book/14781.html)

# Chapter1 - AlphaZeroと機械学習の概要

## AlphaZeroについて
イギリスDeepMind社開発の囲碁・チェス・将棋攻略の人口知能

・AlphaGo (2015/10～）：DeepMind社の囲碁プログラム
・AlphaGo Zero (2017/11) ： AlphaGoの改善版、碁士のデータを使わず
・AlphaZero (2017/12)：AlphaGo Zeroをチェス，囲碁にも対応、盤面の回転による学習データの水増しを廃止

    (1) モンテカルロ木探索→探索の先読みする力
    (2) ディープラーニング→局面の最善手を予測する直感
    (3) 強化学習→自己対戦による経験

  ●論文
  [(PDF) Mastering the game of Go with deep neural networks and tree search (researchgate.net)](https://www.researchgate.net/publication/292074166_Mastering_the_game_of_Go_with_deep_neural_networks_and_tree_search)
  
  [Mastering the game of Go without Human Knowledge | DeepMind](https://deepmind.com/research/publications/mastering-game-go-without-human-knowledge)
  
●動画
[(10) Match 3 - Google DeepMind Challenge Match: Lee Sedol vs AlphaGo - YouTube](https://www.youtube.com/watch?v=qUAmTYHEyM8)

## 畳み込みニューラルネットワーク
AlexNet (2012) : ILSVRC（画像認識競技大会) でトロント大ヒントン教授が開発
GoogleNet (2014) : Googleが開発
ResNet (2015) : Microsoftが開発 

## 強化学習 (Reinforcement learning)
エージェントがある環境の中でその状態に応じてどう行動すると報酬を多くもらえるかを求める手法

エージェント：行動主体
状態：エージェントの行動により変化する環境の要素
報酬：行動の良さを示す指標、行動直後に発生するものを即時報酬、遅れて発生するものを遅延報酬という
収益：報酬の和
価値：状態と方策の条件によって固定的に発生する条件付き報酬のこと
方策：ある状態である行動を起こす確率、強化学習でこの方策を求める
エピソード：学習1回分
ステップ：行動1回分

### 強化学習の目的
　価値の最大化→収益の最大化→報酬を最大化する方策を決める

### 学習サイクル...マルコフ決定過程と呼ばれる
(1) 初期でエージョントはランダムな行動をとる。
(2) 行動後に報酬が発生した場合、(状態、行動、報酬）を経験として記憶する。
(3) 経験に応じて方策をきめる。
(4) 方策にランダムネスを加えて次の行動を決定する。
(5) (2)～(4)を繰り返し将来得られる報酬を最大化する方策を求める。

### 方策を求める手法
・方策反復法
方策に従って行動し成功時の行動を高く評価し方策を更新する。方策勾配法というアルゴリズムがこれにあたる
・価値反復法
次の状態の価値と現在の状態の価値の差分を計算するアルゴリズム、SarsaとQ学習がある

### 探索
ゲーム木：ノードとアークで表すツリー構造
完全ゲーム木：ゲーム開始時にすべての手を含んだゲーム木、ノードの数が多くなり計算ができない。
部分ゲーム木：制限時間内に計算可能な手を含むゲーム木


