from rest_framework.routers import DefaultRouter

from .views import AnalysisJobViewSet

router = DefaultRouter()
router.register("analyses", AnalysisJobViewSet, basename="analysis")

urlpatterns = router.urls
