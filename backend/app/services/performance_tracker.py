import logging
import time
from typing import Dict, List, Optional
from datetime import datetime

from app.models.schemas import PerformanceBenchmark, ReasoningChain, ReasoningStep, PerformanceMetrics, VerificationLevel

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """
    Service for tracking and analyzing QA performance metrics
    Supports demonstrating research efficiency gains vs manual processes
    """
    
    def __init__(self):
        self.benchmarks: Dict[str, PerformanceBenchmark] = {}
        self.reasoning_chains: Dict[str, ReasoningChain] = {}
        self.complexity_estimates = {
            # Estimated manual research times by question complexity
            1: 5.0,   # Simple: 5 minutes
            2: 15.0,  # Medium: 15 minutes  
            3: 30.0,  # Complex: 30 minutes
            4: 60.0,  # Very Complex: 1 hour
            5: 120.0  # Extremely Complex: 2 hours
        }
    
    def estimate_question_complexity(self, question: str) -> int:
        """
        Estimate question complexity on a scale of 1-5
        Based on question length, financial concepts, and complexity indicators
        """
        complexity_indicators = {
            # Financial complexity keywords
            'complex_terms': ['derivative', 'hedge', 'swap', 'option', 'warrant', 'convertible', 'covenant'],
            'analysis_terms': ['trend', 'compare', 'analyze', 'evaluate', 'assess', 'forecast', 'predict'],
            'multi_period': ['quarter', 'annual', 'year-over-year', 'historical', 'timeline'],
            'regulatory': ['SEC', 'GAAP', 'IFRS', 'regulation', 'compliance', 'filing'],
            'comparative': ['versus', 'compared to', 'relative to', 'benchmark', 'peer'],
        }
        
        question_lower = question.lower()
        complexity_score = 1
        
        # Length-based complexity
        if len(question) > 200:
            complexity_score += 1
        elif len(question) > 100:
            complexity_score += 0.5
            
        # Multiple questions or sub-parts
        if '?' in question[:-1]:  # Multiple question marks (excluding the last one)
            complexity_score += 1
            
        # Financial complexity keywords
        for category, terms in complexity_indicators.items():
            matches = sum(1 for term in terms if term in question_lower)
            if matches >= 3:
                complexity_score += 2
            elif matches >= 2:
                complexity_score += 1
            elif matches >= 1:
                complexity_score += 0.5
                
        # Multiple entities or time periods
        if 'and' in question_lower and ('company' in question_lower or 'corporation' in question_lower):
            complexity_score += 1
            
        return min(5, max(1, int(complexity_score)))
    
    def start_reasoning_chain(self, question_id: str, question: str, session_id: str) -> ReasoningChain:
        """Start tracking reasoning chain for a question"""
        reasoning_chain = ReasoningChain(
            question_id=question_id,
            question=question,
            reasoning_steps=[],
            total_duration_ms=0,
            final_confidence=0.0,
            session_id=session_id
        )
        self.reasoning_chains[question_id] = reasoning_chain
        return reasoning_chain
    
    def add_reasoning_step(self, question_id: str, description: str, action_type: str, 
                          sources_consulted: List[str] = None, confidence: float = 0.5,
                          output: str = "", metadata: Dict = None) -> int:
        """Add a reasoning step to the chain"""
        if question_id not in self.reasoning_chains:
            logger.warning(f"Reasoning chain not found for question {question_id}")
            return 0
            
        chain = self.reasoning_chains[question_id]
        step_start = time.time()
        
        step = ReasoningStep(
            step_number=len(chain.reasoning_steps) + 1,
            description=description,
            action_type=action_type,
            sources_consulted=sources_consulted or [],
            confidence=confidence,
            duration_ms=0,  # Will be updated when step completes
            output=output,
            metadata=metadata or {}
        )
        
        # Store start time for duration calculation
        step.metadata['start_time'] = step_start
        chain.reasoning_steps.append(step)
        
        return step.step_number
    
    def complete_reasoning_step(self, question_id: str, step_number: int, 
                              output: str = "", confidence: float = None):
        """Complete a reasoning step and calculate duration"""
        if question_id not in self.reasoning_chains:
            return
            
        chain = self.reasoning_chains[question_id]
        if step_number <= len(chain.reasoning_steps):
            step = chain.reasoning_steps[step_number - 1]
            start_time = step.metadata.get('start_time', time.time())
            step.duration_ms = int((time.time() - start_time) * 1000)
            
            if output:
                step.output = output
            if confidence is not None:
                step.confidence = confidence
                
            # Update total duration
            chain.total_duration_ms = sum(s.duration_ms for s in chain.reasoning_steps)
    
    def finalize_reasoning_chain(self, question_id: str, final_confidence: float) -> Optional[ReasoningChain]:
        """Finalize the reasoning chain"""
        if question_id not in self.reasoning_chains:
            return None
            
        chain = self.reasoning_chains[question_id]
        chain.final_confidence = final_confidence
        chain.total_duration_ms = sum(s.duration_ms for s in chain.reasoning_steps)
        
        return chain
    
    def create_performance_benchmark(self, question_id: str, question: str, 
                                   processing_time_seconds: float, source_count: int,
                                   accuracy_score: float, confidence_score: float,
                                   verification_level: VerificationLevel, session_id: str) -> PerformanceBenchmark:
        """Create a performance benchmark for a completed QA operation"""
        
        complexity_score = self.estimate_question_complexity(question)
        estimated_manual_time = self.complexity_estimates[complexity_score]
        ai_processing_time = processing_time_seconds / 60.0  # Convert to minutes
        
        # Calculate efficiency gain
        efficiency_gain = ((estimated_manual_time - ai_processing_time) / estimated_manual_time) * 100
        efficiency_gain = max(0, efficiency_gain)  # Ensure non-negative
        
        benchmark = PerformanceBenchmark(
            question_id=question_id,
            question=question,
            complexity_score=complexity_score,
            estimated_manual_time=estimated_manual_time,
            ai_processing_time=ai_processing_time,
            efficiency_gain=efficiency_gain,
            source_count=source_count,
            accuracy_score=accuracy_score,
            confidence_score=confidence_score,
            verification_level=verification_level,
            session_id=session_id,
            metadata={
                'processing_time_seconds': processing_time_seconds,
                'complexity_factors': self._analyze_complexity_factors(question)
            }
        )
        
        self.benchmarks[question_id] = benchmark
        logger.info(f"Performance benchmark created for question {question_id}: "
                   f"{efficiency_gain:.1f}% efficiency gain, "
                   f"{ai_processing_time:.2f}min vs {estimated_manual_time:.1f}min estimated manual time")
        
        return benchmark
    
    def get_session_metrics(self, session_id: str) -> PerformanceMetrics:
        """Get aggregated performance metrics for a session"""
        session_benchmarks = [b for b in self.benchmarks.values() if b.session_id == session_id]
        
        if not session_benchmarks:
            return PerformanceMetrics(
                total_questions=0,
                average_efficiency_gain=0,
                average_accuracy_score=0,
                average_processing_time=0,
                complexity_breakdown={},
                time_saved_minutes=0,
                session_id=session_id
            )
        
        total_questions = len(session_benchmarks)
        avg_efficiency = sum(b.efficiency_gain for b in session_benchmarks) / total_questions
        avg_accuracy = sum(b.accuracy_score for b in session_benchmarks) / total_questions
        avg_processing_time = sum(b.ai_processing_time for b in session_benchmarks) / total_questions
        
        complexity_breakdown = {}
        for b in session_benchmarks:
            complexity_breakdown[b.complexity_score] = complexity_breakdown.get(b.complexity_score, 0) + 1
        
        time_saved = sum(b.estimated_manual_time - b.ai_processing_time for b in session_benchmarks)
        
        return PerformanceMetrics(
            total_questions=total_questions,
            average_efficiency_gain=avg_efficiency,
            average_accuracy_score=avg_accuracy,
            average_processing_time=avg_processing_time,
            complexity_breakdown=complexity_breakdown,
            time_saved_minutes=time_saved,
            session_id=session_id
        )
    
    def _analyze_complexity_factors(self, question: str) -> Dict[str, bool]:
        """Analyze what makes a question complex for transparency"""
        question_lower = question.lower()
        
        factors = {
            'long_question': len(question) > 100,
            'multiple_questions': '?' in question[:-1],
            'financial_jargon': any(term in question_lower for term in 
                                  ['derivative', 'hedge', 'covenant', 'warrant']),
            'analysis_required': any(term in question_lower for term in
                                   ['analyze', 'compare', 'evaluate', 'assess']),
            'multi_period': any(term in question_lower for term in
                              ['historical', 'trend', 'year-over-year']),
            'regulatory_focus': any(term in question_lower for term in
                                  ['SEC', 'GAAP', 'regulation', 'compliance']),
            'comparative_analysis': any(term in question_lower for term in
                                      ['versus', 'compared to', 'benchmark'])
        }
        
        return factors
    
    def get_reasoning_chain(self, question_id: str) -> Optional[ReasoningChain]:
        """Get the reasoning chain for a question"""
        return self.reasoning_chains.get(question_id)
    
    def get_performance_benchmark(self, question_id: str) -> Optional[PerformanceBenchmark]:
        """Get the performance benchmark for a specific question"""
        return self.benchmarks.get(question_id)

# Global instance
performance_tracker = PerformanceTracker()
