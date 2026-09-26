 # Projeto Esportivo de Dados

Esta é uma atividade extensionista para o curso de ciência de dados da Uninter.

## Pipeline atual de Análise de Vídeo

* Etapa 0 — Seleção Manual: Definição inicial das caixas de referência (Time 1, Time 2, Goleiros, Árbitro e exclusões) através de cliques em um único frame.
* Etapa 1 — Detecção: Identificação de jogadores utilizando modelo YOLO26 e da bola utilizando YOLO26 ou RT-ETR.
* Etapa 2 — Tracking: Atribuição e manutenção de IDs temporários entre os frames usando algoritmos de rastreamento com diferentes abordagens de geometria e reidentificação (ByteTrack, BoT-SORT, OC-SORT, StrongSORT, NorFair, entre outros).
* Etapa 3 — Associação Geométrica: Conexão dos IDs do rastreador aos IDs lógicos fixos definidos na Etapa 0, baseada em IoU (Interseção sobre União) e distância normalizada.
* Etapa 4 — Classificação de Papel: Identificação da equipe ou função do alvo utilizando métodos de análise de aparência (HSV, K-Means Duplo, MobileNet, SigLIP, DINOv2).
* Etapa 5 — Avaliação de Métricas: Geração de resultados comparativos (FPS, trocas de ID, presença da bola) isolados para cada combinação testada.


### **Próximos Passos:**  
Utilizar modelos treinados com videos no mesmo padrão das filmagens para melhorar o tracking.
Implementar a captura de estatísticas básicas como posse de bola e passes completos.


## Métricas de Desempenho utilizadas nos testes para identificar melhores soluções

**Cobertura**: Mede a porcentagem de tempo em que o sistema detectou com sucesso um jogador que deveria estar visível. Valores elevados (ex: 99%) indicam um rastreamento contínuo e estável, sem perdas de alvo.

**Confirmação de Papel**: Mede a taxa de acerto contínuo na atribuição do time correto. Índices altos provam a eficácia da lógica de "inércia", que avalia o histórico de frames para evitar que um jogador troque de time temporariamente devido a sombras passageiras.

**Divergência de Papel**: Representa a taxa de erro visual (ex: classificar um jogador do Time 1 como Time 2). Esse erro é gerado quando os algoritmos de aparência falham em decorrência de condições adversas.

**Presença da bola**: Quanto a bola estava ativamente sendo reconhecida durante o vídeo.

**FPS**: A velocidade em que a analise foi realizada.

**Frames/Troca**: Quantidade de Frames que rodaram por Quantas trocas de ID aconteceram.

**Troca ID**: Quantidade de vezes que um ID desapareceu.
