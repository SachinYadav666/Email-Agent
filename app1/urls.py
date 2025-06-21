from django.urls import path
from .views import EmailAgentAPIView

urlpatterns = [
    path('api/process-email/', EmailAgentAPIView.as_view()),
]