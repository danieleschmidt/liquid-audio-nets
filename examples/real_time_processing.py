#!/usr/bin/env python3
"""
Real-time audio processing example with Liquid Neural Networks.

Demonstrates continuous audio stream processing with always-on detection,
power management, and performance monitoring.
"""

import numpy as np
import time
import threading
import queue
import sys
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from collections import deque

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.liquid_audio_nets import LNN, AdaptiveConfig

@dataclass
class AudioFrame:
    """Audio frame with metadata."""
    data: np.ndarray
    timestamp: float
    frame_id: int
    sample_rate: int = 16000

class AudioStreamSimulator:
    """Simulates continuous audio stream for demonstration."""
    
    def __init__(self, sample_rate: int = 16000, frame_size: int = 512):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.running = False
        self.frame_counter = 0
        
        # Audio patterns for simulation
        self.patterns = {
            'silence': lambda t: 0.01 * np.random.randn(len(t)),
            'noise': lambda t: 0.1 * np.random.randn(len(t)),
            'wake_word': lambda t: 0.4 * (np.sin(2*np.pi*800*t) + np.sin(2*np.pi*1200*t)) + 0.02*np.random.randn(len(t)),
            'stop_word': lambda t: 0.3 * np.sin(2*np.pi*600*t) * np.exp(-t*2) + 0.02*np.random.randn(len(t)),
            'speech': lambda t: 0.2 * np.sin(2*np.pi*np.cumsum(400 + 200*np.sin(2*np.pi*t*3))) + 0.03*np.random.randn(len(t))
        }
        
        # Scenario sequence
        self.scenario_schedule = [
            (0, 2, 'silence'),      # 0-2s: silence
            (2, 3, 'wake_word'),    # 2-3s: wake word
            (3, 5, 'speech'),       # 3-5s: speech activity
            (5, 6, 'stop_word'),    # 5-6s: stop word  
            (6, 8, 'noise'),        # 6-8s: background noise
            (8, 10, 'silence'),     # 8-10s: silence
        ]
        
    def generate_frame(self, current_time: float) -> AudioFrame:
        """Generate audio frame based on current scenario."""
        # Determine current audio pattern
        pattern = 'silence'
        for start, end, pattern_name in self.scenario_schedule:
            if start <= current_time < end:
                pattern = pattern_name
                break
        
        # Generate frame
        t = np.linspace(0, self.frame_size / self.sample_rate, self.frame_size)
        audio_data = self.patterns[pattern](t).astype(np.float32)
        
        frame = AudioFrame(
            data=audio_data,
            timestamp=current_time,
            frame_id=self.frame_counter,
            sample_rate=self.sample_rate
        )
        
        self.frame_counter += 1
        return frame
    
    def start_stream(self, frame_callback: Callable[[AudioFrame], None], duration: float = 10.0):
        """Start streaming audio frames."""
        self.running = True
        start_time = time.time()
        
        while self.running and (time.time() - start_time) < duration:
            current_time = time.time() - start_time
            frame = self.generate_frame(current_time)
            
            frame_callback(frame)
            
            # Sleep to maintain real-time rate
            time.sleep(self.frame_size / self.sample_rate)
    
    def stop_stream(self):
        """Stop audio streaming."""
        self.running = False

class PerformanceMonitor:
    """Monitor and track processing performance."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.processing_times = deque(maxlen=window_size)
        self.power_consumption = deque(maxlen=window_size)
        self.detection_events = []
        self.frame_count = 0
        
    def record_frame(self, processing_time: float, power_mw: float, detection_result: Dict[str, Any]):
        """Record performance metrics for a frame."""
        self.processing_times.append(processing_time)
        self.power_consumption.append(power_mw)
        self.frame_count += 1
        
        if detection_result.get('keyword_detected', False):
            self.detection_events.append({
                'frame': self.frame_count,
                'time': time.time(),
                'keyword': detection_result.get('keyword'),
                'confidence': detection_result.get('confidence')
            })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time_ms': np.mean(self.processing_times),
            'max_processing_time_ms': np.max(self.processing_times),
            'avg_power_mw': np.mean(self.power_consumption),
            'max_power_mw': np.max(self.power_consumption),
            'frames_processed': self.frame_count,
            'detections_count': len(self.detection_events),
            'real_time_factor': (self.frame_count * 512 / 16000) / (len(self.processing_times) * np.mean(self.processing_times) / 1000) if self.processing_times else 0
        }

class RealTimeProcessor:
    """Real-time audio processing with LNN."""
    
    def __init__(self):
        self.lnn = LNN()
        self.performance_monitor = PerformanceMonitor()
        self.audio_queue = queue.Queue(maxsize=10)  # Small buffer for real-time processing
        self.result_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
        
        # Configure adaptive processing
        adaptive_config = AdaptiveConfig(
            min_timestep=0.003,    # 3ms minimum for real-time
            max_timestep=0.030,    # 30ms maximum  
            energy_threshold=0.08,
            complexity_metric="spectral_flux"
        )
        self.lnn.set_adaptive_config(adaptive_config)
        
        # Alert thresholds
        self.power_alert_threshold = 2.0  # mW
        self.latency_alert_threshold = 20.0  # ms
        
    def start_processing(self):
        """Start the processing thread."""
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop_processing(self):
        """Stop the processing thread."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
    
    def process_frame(self, frame: AudioFrame):
        """Queue audio frame for processing."""
        try:
            self.audio_queue.put_nowait(frame)
        except queue.Full:
            print("⚠️ Audio queue full - dropping frame")
    
    def _processing_loop(self):
        """Main processing loop running in separate thread."""
        while self.running:
            try:
                # Get frame with timeout
                frame = self.audio_queue.get(timeout=0.1)
                
                # Process frame
                start_time = time.perf_counter()
                
                # Keyword detection
                keyword_result = self.lnn.process(frame.data)
                
                # Voice activity detection
                vad_result = self.lnn.detect_activity(frame.data)
                
                processing_time_ms = (time.perf_counter() - start_time) * 1000
                
                # Combine results
                combined_result = {
                    **keyword_result,
                    'vad': vad_result,
                    'processing_time_ms': processing_time_ms,
                    'frame_id': frame.frame_id,
                    'timestamp': frame.timestamp
                }
                
                # Record performance
                self.performance_monitor.record_frame(
                    processing_time_ms,
                    keyword_result['power_mw'],
                    keyword_result
                )
                
                # Check alerts
                self._check_alerts(combined_result)
                
                # Queue result
                self.result_queue.put_nowait(combined_result)
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Processing error: {e}")
    
    def _check_alerts(self, result: Dict[str, Any]):
        """Check for performance alerts."""
        if result['power_mw'] > self.power_alert_threshold:
            print(f"🔋 Power alert: {result['power_mw']:.2f}mW (threshold: {self.power_alert_threshold}mW)")
        
        if result['processing_time_ms'] > self.latency_alert_threshold:
            print(f"⏱️ Latency alert: {result['processing_time_ms']:.2f}ms (threshold: {self.latency_alert_threshold}ms)")
    
    def get_results(self) -> Optional[Dict[str, Any]]:
        """Get latest processing result."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

def demo_real_time_processing():
    """Demonstrate real-time audio processing."""
    print("🎙️ Real-Time Audio Processing Demo")
    print("=" * 60)
    print("Simulating continuous audio stream with keyword detection...")
    
    # Initialize components
    processor = RealTimeProcessor()
    audio_simulator = AudioStreamSimulator()
    
    # Start processing
    processor.start_processing()
    
    print("\n📊 Live Processing Status:")
    print("Time(s) | Audio Type | Keyword | Confidence | Power(mW) | Latency(ms) | VAD")
    print("-" * 80)
    
    # Process results in main thread
    def display_results():
        last_display_time = 0
        while audio_simulator.running:
            result = processor.get_results()
            if result:
                current_time = result['timestamp']
                
                # Display at 2Hz rate
                if current_time - last_display_time >= 0.5:
                    keyword = result.get('keyword', 'None')
                    confidence = result.get('confidence', 0)
                    power = result['power_mw']
                    latency = result['processing_time_ms']
                    vad_speech = "Yes" if result['vad']['is_speech'] else "No"
                    
                    # Determine audio type from scenario
                    audio_type = "silence"
                    for start, end, pattern in audio_simulator.scenario_schedule:
                        if start <= current_time < end:
                            audio_type = pattern
                            break
                    
                    print(f"{current_time:6.1f}s | {audio_type:10} | {keyword:7} | {confidence:10.3f} | {power:8.2f} | {latency:10.2f} | {vad_speech:3}")
                    last_display_time = current_time
            
            time.sleep(0.1)
    
    # Start result display thread
    display_thread = threading.Thread(target=display_results)
    display_thread.daemon = True
    display_thread.start()
    
    # Stream audio
    try:
        audio_simulator.start_stream(processor.process_frame, duration=10.0)
    except KeyboardInterrupt:
        print("\n⏹️ Stopping...")
    
    # Stop processing
    audio_simulator.stop_stream()
    processor.stop_processing()
    
    # Display final statistics
    stats = processor.performance_monitor.get_stats()
    if stats:
        print("\n📈 Performance Summary:")
        print("-" * 40)
        print(f"Frames processed:     {stats['frames_processed']}")
        print(f"Keywords detected:    {stats['detections_count']}")
        print(f"Avg processing time:  {stats['avg_processing_time_ms']:.2f}ms")
        print(f"Max processing time:  {stats['max_processing_time_ms']:.2f}ms")
        print(f"Avg power consumption: {stats['avg_power_mw']:.2f}mW")
        print(f"Max power consumption: {stats['max_power_mw']:.2f}mW")
        print(f"Real-time factor:     {stats['real_time_factor']:.2f}x")
        
        # Performance evaluation
        real_time_ok = stats['real_time_factor'] > 1.0
        power_ok = stats['avg_power_mw'] < 1.5
        latency_ok = stats['avg_processing_time_ms'] < 15.0
        
        print(f"\n✅ Real-time performance: {'PASS' if real_time_ok else 'FAIL'}")
        print(f"✅ Power efficiency:      {'PASS' if power_ok else 'FAIL'}")
        print(f"✅ Low latency:           {'PASS' if latency_ok else 'FAIL'}")
        
        if all([real_time_ok, power_ok, latency_ok]):
            print("\n🎉 All performance targets met!")
        else:
            print("\n⚠️ Some performance targets not met - consider optimization")

def main():
    """Run real-time processing demonstration."""
    print("🚀 Liquid Audio Networks - Real-Time Processing Demo")
    print("This demo shows continuous audio processing capabilities\n")
    
    try:
        demo_real_time_processing()
        
        print("\n\n✅ Real-time demo completed!")
        print("\n💡 Key Features Demonstrated:")
        print("   • Continuous audio stream processing")
        print("   • Multi-threaded real-time architecture")
        print("   • Combined keyword detection + voice activity detection")
        print("   • Performance monitoring and alerts")
        print("   • Adaptive power optimization")
        print("   • Queue-based frame buffering")
        
        print("\n🔧 Production Considerations:")
        print("   • Use proper audio drivers (PyAudio, sounddevice)")
        print("   • Implement circular buffers for embedded targets")
        print("   • Add audio preprocessing (noise reduction, AGC)")
        print("   • Configure interrupt-driven processing")
        print("   • Implement proper error recovery")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())