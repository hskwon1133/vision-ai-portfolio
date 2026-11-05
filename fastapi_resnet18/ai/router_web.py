# 동기식 HTML 라우터
# 동기식 이미지 업로드 및 분류를 처리하며, HTML 템플릿을 렌더링하고 동기식 업로드 페이지를 제공한다.
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from .config import BASE_DIR

router = APIRouter(tags=["실습 화면"])
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@router.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/vision/sync/")

@router.get("/vision/sync/")
def sync_page(request: Request):
    return templates.TemplateResponse(request=request, name="sync.html")

@router.get("/vision/async/")
def async_page(request: Request):
    return templates.TemplateResponse(request=request, name="async.html")