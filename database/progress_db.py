# ai/database/progress_db.py
from datetime import datetime
from typing import Optional, List, Dict
from tinydb import TinyDB, Query, where
from tinydb.table import Document
import json

class ProgressDB:
    """
    Track study progress and interactions using TinyDB
    Features:
    - Log study sessions
    - Record Q&A history
    - Track streaks and goals
    - Export/import data
    """
    
    def __init__(self, db_path: str = "ai/database/progress.json"):
        self.db = TinyDB(db_path, indent=4)
        self.sessions = self.db.table('study_sessions')
        self.interactions = self.db.table('interactions')
        self.goals = self.db.table('goals')
        
    def log_study_session(
        self,
        topic: str,
        duration_min: int,
        resources: Optional[List[str]] = None,
        effectiveness: Optional[int] = None
    ) -> int:
        """Record a study session"""
        doc_id = self.sessions.insert({
            'timestamp': datetime.now().isoformat(),
            'topic': topic,
            'duration_min': duration_min,
            'resources': resources or [],
            'effectiveness': effectiveness,  # 1-5 scale
            'tags': self._generate_tags(topic)
        })
        return doc_id

    def log_interaction(
        self,
        query: str,
        response: str,
        interaction_type: str
    ) -> int:
        """Record Q&A or wellness interactions"""
        return self.interactions.insert({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response[:500],  # Truncate long responses
            'type': interaction_type,
            'sentiment': self._analyze_sentiment(response)
        })

    def get_streak(self, days: int = 7) -> int:
        """Calculate current study streak"""
        recent_sessions = self.sessions.search(
            where('timestamp') >= (datetime.now() - timedelta(days=days)).isoformat()
        )
        return len({s['timestamp'][:10] for s in recent_sessions})  # Unique days

    def get_study_summary(self, period: str = "week") -> Dict:
        """Get summary stats for time period"""
        if period == "week":
            cutoff = datetime.now() - timedelta(days=7)
        else:  # month
            cutoff = datetime.now() - timedelta(days=30)
            
        sessions = self.sessions.search(where('timestamp') >= cutoff.isoformat())
        
        return {
            'total_hours': sum(s['duration_min'] for s in sessions) / 60,
            'top_topics': self._get_top_topics(sessions),
            'streak': self.get_streak()
        }

    def export_data(self, filepath: str) -> None:
        """Export all data to JSON"""
        data = {
            'sessions': self.sessions.all(),
            'interactions': self.interactions.all(),
            'goals': self.goals.all()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    # Private helpers
    def _generate_tags(self, topic: str) -> List[str]:
        """Auto-generate tags from topic"""
        topic = topic.lower()
        tags = []
        if 'math' in topic:
            tags.append('mathematics')
        if 'physics' in topic:
            tags.append('science')
        return tags or ['general']

    def _analyze_sentiment(self, text: str) -> Optional[str]:
        """Basic sentiment analysis"""
        text = text.lower()
        positive_words = ['great', 'good', 'excellent']
        negative_words = ['stress', 'anxious', 'hard']
        
        if any(word in text for word in positive_words):
            return 'positive'
        elif any(word in text for word in negative_words):
            return 'negative'
        return None

    def _get_top_topics(self, sessions: List[Document]) -> List[Dict]:
        """Get most studied topics"""
        from collections import defaultdict
        topic_counts = defaultdict(int)
        for s in sessions:
            topic_counts[s['topic']] += s['duration_min']
        return sorted(topic_counts.items(), key=lambda x: -x[1])[:3]  # Top 3

# Test cases
if __name__ == "__main__":
    db = ProgressDB(":memory:")  # In-memory DB for testing
    
    # Test logging
    db.log_study_session("Linear Algebra", 45, ["textbook.pdf"])
    db.log_interaction("What is a matrix?", "A rectangular array...", "qa")
    
    # Test queries
    print(f"Current streak: {db.get_streak()} days")
    print("Weekly summary:", db.get_study_summary())