# -*- coding: utf-8 -*-
"""Email Agent API Views
"""

import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from app1.utils.agent_utils import (
    create_email_agent_graph
)

# Configure logging
logger = logging.getLogger(__name__)



# Create the graph instance
email_agent = create_email_agent_graph()

class EmailAgentAPIView(APIView):
    """API View for processing emails through the agent."""
    
    def post(self, request):
        """Process an incoming email request."""
        logger.debug("EmailAgentAPIView.post called")
        
        try:
            initial_email = request.data.get('email_content')
            
            if not initial_email:
                logger.warning("No email content provided in request")
                return Response({
                    'error': 'No email content provided'
                }, status=status.HTTP_400_BAD_REQUEST)

            # Initialize state
            initial_state = {
                "initial_email": initial_email,
                "email_category": "",
                "draft_email": "",
                "final_email": "",
                "research_info": [],
                "info_needed": False,
                "num_steps": 0,
                "draft_email_feedback": {}
            }

            # Run the agent
            logger.info(f"Processing email through agent: {initial_email[:100]}...")
            final_state = email_agent.invoke(initial_state)
            logger.info("Email processing completed successfully")

            return Response({
                'status': 'success',
                'email_category': final_state['email_category'],
                'final_email': final_state['final_email'],
                'research_info': final_state['research_info'],
                'num_steps': final_state['num_steps']
            })

        except Exception as e:
            logger.error(f"Error processing email: {str(e)}", exc_info=True)
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)