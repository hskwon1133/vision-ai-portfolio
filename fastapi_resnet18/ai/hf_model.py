# hugging face 모델 로딩을 위한 모듈 파일

from functools import lru_cache

from .config import MODEL_PATH

@lru_cache(maxsize=1)
def get_classifier():
    """서버/워커 프로세스별 최초 요청 때 모델을 로드하고 재사용한다."""
    from transformers import pipeline

    return pipeline("image-classification", model=MODEL_PATH, device=-1)

# 궁금한 점?
# 1. functools.lru_cache는 무엇인가요?
# functools.lru_cache는 Python의 내장 데코레이터로, 함수의 결과를 캐싱하여 동일한 입력에 대해 함수가 다시 호출될 때 이전 결과를 반환하도록 합니다. 
# 이를 통해 성능을 향상시키고, 불필요한 계산을 줄일수 있습니다. lru_cache는 "Least Recently Used" 캐싱 전략을 사용하여, 가장 오래된 캐시 항목을 제거하고 새로운 항목을 추가합니다.

# 2. cache는 무엇이고, 왜 사용하는가?
# cache는 함수의 결과를 저장하는 메모리 공간입니다. 
# 이를 사용하면 동일한 입력에 대해 함수가 다시 호출될 때 이전 결과를 반환하여 성능을 향상시킬 수 있습니다. 
# 특히, 모델 로딩과 같은 비용이 큰 작업을 반복적으로 수행할 때 유용합니다.

# 3. get_classifier() 함수는 무엇을 하는가?
# get_classifier() 함수는 서버/워커 프로세스별 최초 요청 때 모델을 로드하고 재사용하는 역할을 합니다. 
# 이 함수는 transformers 라이브러리의 pipeline을 사용하여 이미지 분류 모델을 로드하고, 이후 동일한 모델을 재사용할 수 있도록 캐싱합니다. 
# 이를 통해 모델 로딩 시간을 줄이고, 서버의 성능을 향상시킬 수 있습니다.

# 4. pipeline은 무엇인가요?
# pipeline은 Hugging Face의 Transformers 라이브러리에서 제공하는 고수준 API로, 다양한 NLP 및 CV 작업을 쉽게 수행할 수 있도록 도와줍니다.

# 5. device=-1은 무엇을 의미하나요?
# device=-1은 모델이 CPU에서 실행되도록 지정하는 옵션입니다. 

# 6. transformers 라이브러리는 무엇인가요?
# transformers 라이브러리는 Hugging Face에서 제공하는 오픈소스 라이브러리로, 다양한 사전 학습된 모델을 쉽게 사용할 수 있도록 도와줍니다.
