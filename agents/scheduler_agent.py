# ai/agents/scheduler_agent.py
from datetime import datetime, timedelta
from typing import List, Dict, Literal
from ai.chains.base_chain import BaseChain
from pydantic import BaseModel

class StudyTask(BaseModel):
    topic: str
    duration: int  # in minutes
    priority: Literal["high", "medium", "low"]
    deadline: datetime

class SchedulerAgent(BaseChain):
    """
    Dynamic study planner with:
    - Deadline-aware scheduling
    - Priority-based task allocation
    - Cognitive load balancing
    """
    
    def __init__(self):
        super().__init__()
        self.max_daily_study = 240  # 4 hours/day max
        self.subject_weights = {
            "math": 1.5,    # Hard subjects get more time
            "history": 1.0,
            "coding": 1.3
        }

    def decompose_goal(self, goal: str, total_hours: int) -> List[StudyTask]:
        """Break goal into subtasks using LLM"""
        prompt = f"""Break this study goal into 3-5 subtasks:
        Goal: "{goal}" (Total: {total_hours}h)
        
        Respond as JSON:
        {{"tasks": [{{"topic": "...", "hours": float}}]}}"""
        
        try:
            result = self.llm.invoke(prompt)
            tasks = eval(result)["tasks"]  # Simple parsing (use JSON loader in prod)
            return [
                StudyTask(
                    topic=t["topic"],
                    duration=int(t["hours"] * 60),
                    priority="high" if i == 0 else "medium",  # First task = highest priority
                    deadline=datetime.now() + timedelta(days=3)  # Temp deadline
                )
                for i, t in enumerate(tasks)
            ]
        except Exception as e:
            # Fallback to default tasks
            return [
                StudyTask(
                    topic=goal,
                    duration=120,
                    priority="high",
                    deadline=datetime.now() + timedelta(days=1)
                )
            ]

    def generate_schedule(self, tasks: List[StudyTask], available_days: int) -> Dict[str, List[str]]:
        """Create daily timetable with cognitive load balancing"""
        schedule = {}
        remaining_tasks = tasks.copy()
        
        for day in range(available_days):
            date = (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d")
            schedule[date] = []
            daily_minutes = 0
            
            # Process high-priority tasks first
            for priority in ["high", "medium", "low"]:
                for task in remaining_tasks[:]:  # Iterate copy for safe removal
                    if task.priority == priority:
                        alloc_min = min(
                            task.duration,
                            self.max_daily_study - daily_minutes
                        )
                        
                        if alloc_min > 0:
                            schedule[date].append(
                                f"{task.topic} ({alloc_min}min)"
                            )
                            daily_minutes += alloc_min
                            task.duration -= alloc_min
                            
                            if task.duration <= 0:
                                remaining_tasks.remove(task)
            
            # Add break slots
            if schedule[date]:
                schedule[date].append("15min break")
        
        return schedule

    def plan_study(self, goal: str, days_until_deadline: int) -> Dict:
        """End-to-end planning pipeline"""
        total_hours = min(days_until_deadline * 4, 40)  # Cap at 40h total
        tasks = self.decompose_goal(goal, total_hours)
        return self.generate_schedule(tasks, days_until_deadline)

# Test implementation
if __name__ == "__main__":
    agent = SchedulerAgent()
    
    # Example: Prepare for math exam in 5 days
    plan = agent.plan_study(
        goal="Linear algebra final exam",
        days_until_deadline=5
    )
    
    print("📅 Study Plan:")
    for date, tasks in plan.items():
        print(f"\n{date}:")
        for task in tasks:
            print(f"  - {task}")