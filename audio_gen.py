import pyaudio
import numpy as np
import librosa
import time
import random
from scipy import signal
import os
from datetime import datetime
from pydub import AudioSegment
import wave
import io

# Configure audio input
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100  # Sample rate
CHUNK = 1024  # Buffer size

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Start streaming from microphone
stream_in = audio.open(format=FORMAT, channels=CHANNELS,
                   rate=RATE, input=True,
                   frames_per_buffer=CHUNK)

# Create output stream
stream_out = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, output=True)

# Create output directory if it doesn't exist
output_dir = "ambient_recordings"
os.makedirs(output_dir, exist_ok=True)

def get_audio_input():
    # Capture audio data
    data = np.frombuffer(stream_in.read(CHUNK, exception_on_overflow=False), dtype=np.float32)
    return data

def analyze_audio(audio_data, sample_rate=44100):
    # Reduce n_fft to match input length
    n_fft = min(2048, len(audio_data))
    
    # Extract audio features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate, n_fft=n_fft)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate, n_fft=n_fft)[0]
    
    # Calculate amplitude (volume)
    rms = np.sqrt(np.mean(np.square(audio_data)))
    
    # Extract pitch information (simplified)
    if len(audio_data) > 0:
        fft_data = np.abs(np.fft.rfft(audio_data))
        freqs = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
        max_idx = np.argmax(fft_data)
        dominant_freq = freqs[max_idx] if max_idx < len(freqs) else 440
    else:
        dominant_freq = 440
    
    return {
        'centroid': np.mean(spectral_centroid) if len(spectral_centroid) > 0 else 1000,
        'rolloff': np.mean(spectral_rolloff) if len(spectral_rolloff) > 0 else 2000,
        'amplitude': rms,
        'dominant_freq': dominant_freq
    }

def map_features_to_music_params(features):
    # Map audio features to musical parameters
    
    # Brightness to chord type (brighter sound = major chords, darker = minor)
    brightness = features['centroid'] / 5000  # Normalize
    is_major = brightness > 0.5
    
    # Volume to note density
    note_density = min(1.0, features['amplitude'] * 10)
    
    # Dominant frequency to base pitch
    # Convert frequency to MIDI note number (approximation)
    if features['dominant_freq'] > 20:  # Audible frequency
        base_pitch = int(12 * np.log2(features['dominant_freq'] / 440) + 69) % 12
    else:
        base_pitch = 0
        
    return {
        'is_major': is_major,
        'note_density': note_density,
        'brightness': brightness,
        'base_pitch': int(base_pitch)
    }

def generate_sine_wave(freq, duration, sample_rate=44100, amplitude=0.5):
    """Generate a sine wave at the given frequency"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)

def note_to_freq(note):
    """Convert MIDI note number to frequency"""
    return 440 * 2**((note - 69) / 12)

def generate_ambient_audio(musical_params, duration=5, sample_rate=44100):
    # Define key and scale
    base_note = 60 + musical_params['base_pitch']  # Middle C + offset
    
    # Create scale (major or minor)
    if musical_params['is_major']:
        intervals = [0, 2, 4, 5, 7, 9, 11]  # Major scale
    else:
        intervals = [0, 2, 3, 5, 7, 8, 10]  # Minor scale
    
    scale = [base_note + i for i in intervals]
    
    # Generate notes based on density
    notes_to_play = max(1, int(musical_params['note_density'] * 5))
    
    # Generate audio
    audio_data = np.zeros(int(sample_rate * duration))
    
    # Generate several notes
    for _ in range(notes_to_play):
        # Select a random note from our scale
        note = random.choice(scale) + random.choice([-12, 0, 0, 12])
        freq = note_to_freq(note)
        
        # Random amplitude and duration for each note
        amp = random.uniform(0.1, 0.3) * musical_params['note_density']
        note_duration = random.uniform(0.5, duration)
        
        # Generate the tone (sine or triangle wave based on brightness)
        if random.random() < musical_params['brightness']:
            # Brighter sound - triangle wave
            t = np.linspace(0, note_duration, int(sample_rate * note_duration), endpoint=False)
            tone = amp * signal.sawtooth(2 * np.pi * freq * t, 0.5)  # 0.5 gives triangle
        else:
            # Darker sound - sine wave
            tone = generate_sine_wave(freq, note_duration, sample_rate, amp)
        
        # Apply fade in/out
        fade_samples = int(sample_rate * 0.1)  # 100ms fade
        if len(tone) > 2 * fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            tone[:fade_samples] *= fade_in
            tone[-fade_samples:] *= fade_out
        
        # Mix into the output at a random position
        start_pos = random.randint(0, max(0, len(audio_data) - len(tone)))
        end_pos = min(start_pos + len(tone), len(audio_data))
        audio_data[start_pos:end_pos] += tone[:end_pos-start_pos]
    
    # Normalize to prevent clipping
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9
    
    return audio_data.astype(np.float32)

def save_audio_as_mp3(audio_buffer, sample_rate):
    """Save the recorded audio buffer as an MP3 file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"ambient_session_{timestamp}")
    wav_filename = f"{filename}.wav"
    mp3_filename = f"{filename}.mp3"
    
    # Convert float32 audio to int16 for WAV
    audio_int16 = (np.concatenate(audio_buffer) * 32767).astype(np.int16)
    
    # Save as WAV first
    with wave.open(wav_filename, 'wb') as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(2)  # 2 bytes for int16
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    # Convert WAV to MP3 using pydub
    audio_segment = AudioSegment.from_wav(wav_filename)
    audio_segment.export(mp3_filename, format="mp3")
    
    # Remove the temporary WAV file
    os.remove(wav_filename)
    
    return mp3_filename

def main():
    # Buffer to store all generated audio for recording
    recording_buffer = []
    
    try:
        print("Starting ambient sound to music generator...")
        print("Press Ctrl+C to stop and save the recording")
        
        while True:
            # Get audio input
            audio_data = get_audio_input()
            
            # Analyze audio features
            features = analyze_audio(audio_data)
            print(f"Detected features: centroid={features['centroid']:.1f}, amplitude={features['amplitude']:.3f}")
            
            # Map to musical parameters
            musical_params = map_features_to_music_params(features)
            key_type = 'major' if musical_params['is_major'] else 'minor'
            print(f"Musical params: key={key_type}, density={musical_params['note_density']:.2f}")
            
            # Generate audio based on the parameters
            generated_audio = generate_ambient_audio(musical_params, duration=2)
            
            # Add to recording buffer
            recording_buffer.append(generated_audio)
            
            # Play the generated audio
            stream_out.write(generated_audio.tobytes())
            
    except KeyboardInterrupt:
        print("\nStopping and saving the recording...")
        
        # Save the entire session
        if recording_buffer:
            mp3_file = save_audio_as_mp3(recording_buffer, RATE)
            print(f"Saved session to: {mp3_file}")
    finally:
        # Clean up
        stream_in.stop_stream()
        stream_in.close()
        stream_out.stop_stream()
        stream_out.close()
        audio.terminate()

if __name__ == "__main__":
    main()