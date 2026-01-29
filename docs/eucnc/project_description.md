# Projeto inMotion

O projeto inMotion (Integrated AI Platform for Urban Mobility and Transportation) surge como uma iniciativa estratégica de Investigação e Desenvolvimento (I&D) em Copromoção, liderada pela Wavecom em colaboração com o Instituto de Telecomunicações (IT). O projeto foca-se nos desafios críticos da Mobilidade Urbana Sustentável e Inteligente, num cenário onde mais de 75% dos cidadãos da UE residem em cidades e os sistemas de transporte ineficientes contribuem para 24% das emissões de gases de efeito estufa.
A solução proposta incide sobre os mecanismos de gestão de Sistemas Inteligentes de Transportes (ITS) e plataformas de Mobilidade como Serviço (MaaS). O objetivo central é superar as limitações dos sistemas atuais de bilhética e contagem de passageiros, que não permitem uma caracterização detalhada e contínua do percurso do passageiro (origem-destino) de forma não intrusiva.
O projeto estrutura-se em dois Produtos/Serviços principais:
PPS1 - Sistema de Deteção e Tracking de Passageiros: Focado na recolha anónima de dados através de redes sem fios (Wi-Fi e Bluetooth) e fusão sensorial.
PPS2 - Plataforma de IA para Mobilidade Urbana: Uma camada de inteligência que utiliza Redes Neuronais e Processamento de Linguagem Natural (NLP) para otimização de rotas e análise preditiva.

## Maybe not consider

O presente Relatório de Estudos Preliminares (E1.1) constitui o primeiro marco (M1.1) da Atividade 1 e serve como base técnico-científica para validar as abordagens de Investigação Industrial. Os seus objetivos específicos incluem:
Definição de Requisitos: Consolidar os requisitos funcionais e não funcionais para o PPS1 e PPS2 com base em cenários reais de transporte multimodal.

Investigação do Estado da Arte (T1.1): Avaliar tecnologias de deteção e tracking, com especial enfoque na viabilidade de redes federadas (ex: Eduroam, OpenRoaming) para garantir a continuidade do rastreio sem necessidade de associação manual pelo utilizador.

Estudo de Viabilidade de Matrizes OD (T1.3): Analisar a metodologia para converter dados de tracking anónimo em matrizes de Origem-Destino, essenciais para o redimensionamento de redes de transporte.

Análise de Técnicas de IA (T1.4): Estudar a aplicabilidade de modelos de Machine Learning e sistemas RAG (Retrieval-Augmented Generation) para permitir que gestores interajam com os dados via linguagem natural.

Consolidação de TRL: Validar a transição do nível de maturidade tecnológica de TRL 3 para as fases subsequentes de desenvolvimento experimental.

## Metodologia da recolha dos dados e alguma interpretação

Testes e Resultados
Metodologia
A metodologia adotada baseou-se na recolha, em um ambiente controlado, de dados experimentais com o objetivo de simular um cenário real de interação entre passageiros e um veículo de transporte público. Para tal, foi captada a força do sinal (RSSI) entre dispositivos e um ponto de acesso fixo para classificar percursos efetuados.
Configuração do Ambiente de Testes
De modo a simular o ambiente de um autocarro e de uma paragem, foram definidas duas zonas distintas:
Zona A: Utilizou-se uma sala (capacidade de isolamento da sala..?) para simular o interior de um autocarro. O router foi posicionado ao lado da porta, dentro da sala (conforme ilustrado na imagem), como uma possível configuração de instalação do equipamento nos autocarros.
Zona B: O corredor adjacente à sala, que pretende representar uma paragem do autocarro.

Recolha de Dados
A extração dos dados foi realizada em tempo real, através de uma ligação Ethernet entre o ponto de acesso e um computador portátil. Foi utilizado um script em Python de modo a extrair os dados capturados pelo equipamento. Os dados foram extraídos em intervalos de 10 segundos, compostos por 10 recolhas com intervalos de 1 segundo entre cada, permitindo assim capturar a variação à medida que cada indivíduo se desloca. Entre os dados capturados pelo equipamento, destacam-se:
Endereço MAC: Identificação do dispositivo
RSSI: Força do sinal recebido
Tráfego: Volume de dados transmitidos e recebidos (há relevância em mencionar?)
De modo a capturar a variação do sinal RSSI (Received Signal Strength Indicator) em diferentes cenários de movimento, foram utilizados 4 dispositivos distintos de modo a garantir a diversidade de hardware (referir dispositivos?) e foram definidas 4 rotas distintas, de modo a cobrir as transições possíveis:
Permanência no Autocarro (A -> A): Definida por movimentos ou permanência estática dentro da sala.
Permanência na Paragem (B -> B): Definida por movimentos ou permanência estática no corredor adjacente à sala.
Desembarque do Autocarro (A -> B): Transição de dentro da sala para o corredor.
Embarque no Autocarro (B -> A): Transição do corredor para dentro da sala.
Controlo de Ruído:
A recolha dos dados foi dividida em dois níveis de complexidade:
Recolha Isolada: Estas recolhas foram efetuadas com apenas um dispositivo de cada vez, de modo a fornecer dados mais “limpos”, ou seja, com menor ruído e menos interferências externas. Para esta recolha, foram utilizados apenas dois dos quatro dispositivos, com 20 percursos efetuados, por cada um, para cada rota.
Recolha (..): Em pares, com os 4 dispositivos em simultâneo, foram realizadas as diversas combinações de rotas distintas possíveis. Este cenário introduz um certo nível de ruído de sinal e tem como objetivo simular um ambiente mais realista.
Ao longo da recolha dos dados, de modo a contornar constrangimentos com o equipamento relativamente à
Extração, Limpeza e Estrutura Final dos Dados
Após a recolha, os dados brutos foram processados de modo a transformar as capturas efetuadas a cada segundo, de cada rota, em uma estrutura mais adequada para análise.
Dados Brutos:
Os dados capturados pelo equipamento apresentavam-se num formato de dicionário Python, em que cada captura continha metadados de rede de todos os dispositivos ligados simultaneamente ao ponto de acesso. Uma única captura, das 10 realizadas em cada percurso é do género:
{'phy0-ap0': [], 'phy1-ap0': [{'mac': '[...]', 'rssi': -62, 'rx_bytes': […], 'tx_bytes': […], [...]}, {'mac': [...]}]
Para cada percurso (intervalo de 10 segundos), foram efetuadas 10 capturas, uma cada segundo, destes dados.
Estruturação dos Dados:
O processo de limpeza dos dados focou-se em isolar o comportamento de cada dispositivo - identificado pelo seu endereço MAC - bem como a evolução temporal do sinal de rede correspondente, identificando com um rótulo o percurso efetuado e se a captura foi feita num ambiente ruidoso ou não. O procedimento foi o seguinte:
Agregação Temporal: Para cada dispositivo, as 10 leituras individuais de RSSI foram transpostas de linhas para colunas. Deste modo, cada linha do dataset final passou a representar um percurso completo com os 10 valores de RSSI ao longo deste.
Rotulagem dos Dados:
Identificação dos Percursos: A cada sequência, foi atribuída a classe correspondente ao percurso efetuado (ex: BA para entrada, AB para saída, etc).
Label de Ruído: A cada entrada foi atribuída uma variável booleana (True/False) para identificar se a recolha foi realizada em simultâneo com outros dispositivos (introduzindo ruído de sinal) ou de forma isolada.
Filtragem de Atributos: Dados não essenciais para este estudo, como bytes transmitidos ou connected_time foram descartados, mantendo-se apenas o endereço MAC como identificador e os valores de RSSI como variáveis independentes.
Estrutura Final do Conjunto de Dados (CSV):
O resultado final deste processo é um dataset consolidado em que cada linha corresponde a uma rota efetuada por um dispositivo, rotulada com o percurso efetuado e se foi ou não recolhida em um ambiente ruidoso.
Esta estrutura permite uma análise não apenas da intensidade do sinal, mas da sua variação ao longo dos 10 segundos de percurso, que é crucial para distinguir as diferentes trajetórias efetuadas.
