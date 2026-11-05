#기본 모델 경로·업로드 제한 파일
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
MODEL_PATH = os.getenv("MODEL_PATH", "microsoft/resnet-18")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://vision_user:vision_pass@127.0.0.1:5432/vision_db",
)
REDIS_CACHE_URL = os.getenv("REDIS_CACHE_URL", "redis://127.0.0.1:6379/1")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/2")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/3")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
MEDIA_DIR = BASE_DIR / "media" / "jobs"

## 궁금한 점: 
# 1. dotenv는 무엇인가요?
# dotenv는 Python에서 환경 변수를 쉽게 관리할 수 있도록 도와주는 라이브러리입니다. .env 파일에 환경 변수를 정의하고, 이를 코드에서 불러와 사용할 수 있게 해줍니다.

# 2. dotenv를 사용하여 환경 변수를 로드하는 이유는 무엇인가요? 
# dotenv를 사용하여 환경 변수를 로드하는 이유는 여러 가지가 있습니다:
# 1. **보안**: 환경 변수에 민감한 정보를 저장할 수 있습니다.
# 2. **유연성**: 코드에서 직접 값을 하드코딩하지 않고, 환경 변수로 설정하면 다양한 환경(개발, 테스트, 프로덕션)에서 쉽게 변경할 수 있습니다.

## 궁금한 점: BASE_DIR를 설정하는 이유는 무엇인가요?
# BASE_DIR를 설정하는 이유는 프로젝트의 루트 디렉토리를 기준으로 파일 경로를 쉽게 관리하기 위해서입니다. 
# 이렇게 하면 코드가 다른 환경에서도 일관되게 작동할 수 있습니다.

# 3. MAX_UPLOAD_BYTES와 MAX_IMAGE_PIXELS를 설정하는 이유는 무엇인가요?
# MAX_UPLOAD_BYTES와 MAX_IMAGE_PIXELS를 설정하는 이유는 업로드되는 파일의 크기와 이미지의 해상도를 제한하여 서버의 리소스를 보호하고, 악의적인 사용을 방지하기 위해서입니다. 
# 이를 통해 서버가 과부하에 걸리거나, 너무 큰 파일로 인해 서비스가 중단되는 것을 방지할 수 있습니다.