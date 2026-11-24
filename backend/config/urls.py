from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView

from analyses.page_views import dashboard, index


urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("accounts.urls")),
    path("api/v1/", include("analyses.urls")),
    path("analyses/", index, name="analysis_index"),
    path("", RedirectView.as_view(pattern_name="analysis_index"), name="home"),
    path("dashboard/", dashboard, name="dashboard"),    
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
