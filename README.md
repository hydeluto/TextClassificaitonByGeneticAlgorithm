# TextClassificaitonByGeneticAlgorithm

CNN을 통해 Text를 분류하고 Genetic Algorithm을 통해 parameter를 최적화한다.

* 최적화 대상 parameter
1) character count
2) embedding dimension
3) number of filters
4) filter size
5) batch size
6) learning rate
7) weight decay

* 실행 환경
- CPU : Intel Core i7-1165G7 @ 2.8GHz
- Memory : 16GB

* 데이터
- NSMC (https://github.com/e9t/nsmc)
  
* 실행 방법
- common.py에 parameter들을 setting하고 main.py 실행

* 제시된 실행 환경과 데이터 기준 실행 결과
- 1 epoch 수행시간 기준 24sec로 accuracy 80% 성능 도출
