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
    """Capture audio data from microphone"""
    data = np.frombuffer(stream_in.read(CHUNK, exception_on_overflow=False), dtype=np.float32)
    return data

def analyze_audio(audio_data, sample_rate=44100):
    """Analyze audio features from input"""
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
    """Map audio features to musical parameters"""
    
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

def apply_fade(audio_data, fade_in_time=0.1, fade_out_time=0.1, sample_rate=44100):
    """Apply fade in/out to audio data"""
    fade_in_samples = int(fade_in_time * sample_rate)
    fade_out_samples = int(fade_out_time * sample_rate)
    
    result = np.copy(audio_data)
    
    # Apply fade in
    if fade_in_samples > 0 and len(result) > fade_in_samples:
        fade_in = np.linspace(0, 1, fade_in_samples)
        result[:fade_in_samples] *= fade_in
    
    # Apply fade out
    if fade_out_samples > 0 and len(result) > fade_out_samples:
        fade_out = np.linspace(1, 0, fade_out_samples)
        result[-fade_out_samples:] *= fade_out
    
    return result

def generate_modulated_tone(freq, duration, sample_rate=44100, amp=0.5, wave_type='sine'):
    """Generate a tone with modulated parameters for more interesting timbre"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Add frequency modulation (slight vibrato)
    vibrato_rate = random.uniform(4.0, 7.0)  # Hz
    vibrato_depth = random.uniform(0.005, 0.02)
    freq_mod = freq * (1.0 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t))
    
    # Generate the base waveform
    phase = 2 * np.pi * np.cumsum(freq_mod) / sample_rate
    
    if wave_type == 'sine':
        tone = amp * np.sin(phase)
    elif wave_type == 'triangle':
        tone = amp * signal.sawtooth(phase, 0.5)
    elif wave_type == 'square':
        tone = amp * signal.square(phase)
    elif wave_type == 'sawtooth':
        tone = amp * signal.sawtooth(phase)
    elif wave_type == 'complex':
        # Mix of harmonics for a richer sound
        tone = amp * (0.6 * np.sin(phase) + 
                      0.3 * np.sin(2 * phase) + 
                      0.1 * np.sin(3 * phase))
    
    # Add amplitude modulation (tremolo)
    tremolo_rate = random.uniform(0.5, 4.0)  # Hz
    tremolo_depth = random.uniform(0.1, 0.3)
    amp_mod = 1.0 - tremolo_depth + tremolo_depth * np.sin(2 * np.pi * tremolo_rate * t)
    tone = tone * amp_mod
    
    # Apply fade in/out
    return apply_fade(tone, 0.05, 0.08, sample_rate)

def generate_chord_progression(base_note, is_major):
    """Generate a chord progression based on the key"""
    if is_major:
        # Major key progressions (I-IV-V-I, I-vi-IV-V, etc.)
        progressions = [
            [0, 3, 4, 0],  # I-IV-V-I
            [0, 5, 3, 4],  # I-vi-IV-V
            [0, 3, 5, 4],  # I-IV-vi-V
            [0, 5, 1, 4]   # I-vi-ii-V
        ]
        # Major scale degrees
        chord_types = ['major', 'minor', 'minor', 'major', 'major', 'minor', 'diminished']
    else:
        # Minor key progressions (i-iv-V-i, i-VI-III-V, etc.)
        progressions = [
            [0, 3, 4, 0],  # i-iv-V-i
            [0, 5, 2, 4],  # i-VI-III-V
            [0, 3, 5, 4],  # i-iv-VI-V
            [0, 2, 3, 4]   # i-III-iv-V
        ]
        # Natural minor scale degrees
        chord_types = ['minor', 'diminished', 'major', 'minor', 'minor', 'major', 'major']
    
    # Choose a progression
    progression = random.choice(progressions)
    
    # Define scale degrees
    if is_major:
        scale_degrees = [0, 2, 4, 5, 7, 9, 11]  # Major scale
    else:
        scale_degrees = [0, 2, 3, 5, 7, 8, 10]  # Natural minor scale
    
    # Build chords for the progression
    chords = []
    for degree in progression:
        # Base note of the chord
        root = base_note + scale_degrees[degree % len(scale_degrees)]
        
        # Build the chord based on chord type
        if chord_types[degree % len(chord_types)] == 'major':
            # Major chord: root, major third, perfect fifth
            chords.append([root, root + 4, root + 7])
        elif chord_types[degree % len(chord_types)] == 'minor':
            # Minor chord: root, minor third, perfect fifth
            chords.append([root, root + 3, root + 7])
        elif chord_types[degree % len(chord_types)] == 'diminished':
            # Diminished chord: root, minor third, diminished fifth
            chords.append([root, root + 3, root + 6])
        
    return chords

def generate_rhythmic_patterns(density):
    """Generate different rhythm patterns based on density"""
    if density < 0.3:
        # Sparse rhythm
        patterns = [
            [1, 0, 0, 0, 0.5, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0.7, 0],
            [0.8, 0, 0, 0, 0.6, 0, 0, 0]
        ]
    elif density < 0.6:
        # Medium rhythm
        patterns = [
            [1, 0, 0.6, 0, 0.8, 0, 0.4, 0],
            [0.9, 0, 0.5, 0.3, 0, 0.4, 0, 0.3],
            [0.8, 0.4, 0, 0.5, 0.7, 0, 0.4, 0]
        ]
    else:
        # Dense rhythm
        patterns = [
            [0.8, 0.3, 0.6, 0.2, 0.9, 0.3, 0.7, 0.4],
            [0.9, 0.5, 0.7, 0.4, 0.8, 0.6, 0.5, 0.7],
            [0.7, 0.4, 0.8, 0.3, 0.9, 0.5, 0.7, 0.6]
        ]
    
    # Choose a random pattern based on density
    return random.choice(patterns)

def generate_melodic_pattern(scale, pattern_type):
    """Generate a melodic pattern using the given scale"""
    length = 8  # Standard pattern length
    
    if pattern_type == 'ascending':
        # Simple ascending pattern
        return [scale[i % len(scale)] for i in range(length)]
    elif pattern_type == 'descending':
        # Simple descending pattern
        return [scale[len(scale) - 1 - (i % len(scale))] for i in range(length)]
    elif pattern_type == 'wave':
        # Wave pattern (up and down)
        wave = []
        for i in range(length):
            pos = i % (2 * len(scale))
            if pos < len(scale):
                wave.append(scale[pos])
            else:
                wave.append(scale[2 * len(scale) - pos - 1])
        return wave
    elif pattern_type == 'arpeggio':
        # Arpeggiated pattern (if scale has enough notes)
        if len(scale) >= 5:
            return [scale[0], scale[2], scale[4], scale[2]] * 2
        else:
            return [scale[0], scale[-1], scale[0], scale[-1]] * 2
    else:  # random
        # Random pattern with emphasis on certain scale degrees
        random_pattern = []
        weights = [0.25, 0.1, 0.15, 0.1, 0.2, 0.1, 0.1]  # Root, third, fifth are emphasized
        for _ in range(length):
            scale_idx = random.choices(
                range(min(len(scale), len(weights))), 
                weights=weights[:len(scale)]
            )[0]
            random_pattern.append(scale[scale_idx])
        return random_pattern

def apply_simple_reverb(audio_data, sample_rate=44100, reverb_time=1.0):
    """Apply a simplified reverb effect (efficient approximation)"""
    # Create several delayed copies with decreasing amplitude
    result = np.copy(audio_data)
    
    # Define delay times and amplitudes
    delays = [int(0.05 * sample_rate), int(0.1 * sample_rate), 
              int(0.15 * sample_rate), int(0.2 * sample_rate)]
    amps = [0.4, 0.25, 0.15, 0.1]
    
    # Scale by reverb time
    amps = [a * reverb_time for a in amps]
    
    # Add delayed copies
    for delay, amp in zip(delays, amps):
        if delay < len(audio_data):
            delayed = np.zeros_like(audio_data)
            delayed[delay:] = audio_data[:-delay] * amp
            result += delayed
    
    # Normalize
    if np.max(np.abs(result)) > 0:
        result = result / np.max(np.abs(result)) * 0.9
        
    return result

def generate_ambient_audio(musical_params, duration=5, sample_rate=44100):
    """Generate ambient audio with enhanced melodic and rhythmic features"""
    # Define key and scale
    base_note = 60 + musical_params['base_pitch']  # Middle C + offset
    
    # Create scale (major or minor)
    if musical_params['is_major']:
        intervals = [0, 2, 4, 5, 7, 9, 11]  # Major scale
    else:
        intervals = [0, 2, 3, 5, 7, 8, 10]  # Natural minor scale
    
    scale = [base_note + i for i in intervals]
    
    # Generate a chord progression
    chord_progression = generate_chord_progression(base_note, musical_params['is_major'])
    
    # Generate rhythm pattern
    rhythm = generate_rhythmic_patterns(musical_params['note_density'])
    
    # Choose a melodic pattern type based on brightness
    if musical_params['brightness'] > 0.7:
        pattern_type = 'ascending'
    elif musical_params['brightness'] > 0.5:
        pattern_type = 'wave'
    elif musical_params['brightness'] > 0.3:
        pattern_type = 'arpeggio'
    else:
        pattern_type = 'descending'
    
    # Generate melodic pattern
    melody = generate_melodic_pattern(scale, pattern_type)
    
    # Initialize audio data
    audio_data = np.zeros(int(sample_rate * duration))
    
    # LAYER 1: Bass drone
    # ------------------
    bass_note = base_note - 12  # One octave lower
    bass_freq = note_to_freq(bass_note)
    bass_amp = 0.15 * musical_params['note_density']
    
    # Generate bass with subtle modulation
    bass_tone = generate_modulated_tone(
        bass_freq, duration, sample_rate, 
        amp=bass_amp, 
        wave_type='sine'
    )
    
    # Add to main audio
    audio_data += bass_tone
    
    # LAYER 2: Chord progression
    # --------------------------
    chord_duration = duration / len(chord_progression)
    
    for chord_idx, chord in enumerate(chord_progression):
        chord_start = int(chord_idx * chord_duration * sample_rate)
        
        # Generate each note in the chord
        for note in chord:
            # Convert to frequency
            freq = note_to_freq(note)
            
            # Amplitude based on density
            amp = 0.1 * musical_params['note_density']
            
            # Choose wave type based on brightness
            if musical_params['brightness'] > 0.6:
                wave_type = 'triangle'
            elif musical_params['brightness'] > 0.3:
                wave_type = 'sine'
            else:
                wave_type = 'complex'
            
            # Generate tone with modulation
            tone = generate_modulated_tone(
                freq, chord_duration, sample_rate, 
                amp=amp, 
                wave_type=wave_type
            )
            
            # Add to audio at the right position
            end_pos = min(chord_start + len(tone), len(audio_data))
            audio_data[chord_start:end_pos] += tone[:end_pos-chord_start]
    
    # LAYER 3: Melodic line following rhythm
    # -------------------------------------
    rhythm_steps = len(rhythm)
    step_duration = duration / rhythm_steps
    
    for step_idx in range(rhythm_steps):
        # Only play notes where rhythm is active
        if rhythm[step_idx] > 0 and random.random() < rhythm[step_idx]:
            # Get the note from the melody pattern
            melody_idx = step_idx % len(melody)
            note = melody[melody_idx] + 12  # One octave higher than the base
            
            # Convert to frequency
            freq = note_to_freq(note)
            
            # Note duration based on rhythm strength
            note_amp = 0.13 * rhythm[step_idx]
            note_duration = step_duration * random.uniform(0.7, 1.8)
            
            # Add variation to each note
            if random.random() < 0.3:
                # Occasional octave jump
                freq *= random.choice([0.5, 2.0])
            
            # Choose wave type for variety
            if musical_params['brightness'] > 0.5:
                wave_type = random.choice(['triangle', 'sine'])
            else:
                wave_type = random.choice(['sine', 'complex'])
            
            # Generate tone
            tone = generate_modulated_tone(
                freq, note_duration, sample_rate,
                amp=note_amp,
                wave_type=wave_type
            )
            
            # Add to audio at the right rhythm position
            start_pos = int(step_idx * step_duration * sample_rate)
            end_pos = min(start_pos + len(tone), len(audio_data))
            audio_data[start_pos:end_pos] += tone[:end_pos-start_pos]
    
    # LAYER 4: Ambient texture background (filtered noise)
    # ---------------------------------------------------
    if musical_params['note_density'] > 0.5:
        # Generate noise base
        noise = np.random.randn(int(sample_rate * duration)) * 0.05
        
        # Apply a simple low-pass filter (moving average)
        window_size = int(0.01 * sample_rate)
        window = np.ones(window_size) / window_size
        filtered_noise = np.convolve(noise, window, mode='same')
        
        # Modulate amplitude based on brightness
        noise_amp = 0.05 * musical_params['brightness']
        filtered_noise *= noise_amp
        
        # Add to audio
        audio_data += filtered_noise
    
    # Apply reverb
    reverb_time = 0.8 + 0.4 * musical_params['brightness']
    audio_data = apply_simple_reverb(audio_data, sample_rate, reverb_time)
    
    # Final normalization to prevent clipping
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
        print("Starting enhanced ambient sound to music generator...")
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
            print(f"Musical params: key={key_type}, density={musical_params['note_density']:.2f}, brightness={musical_params['brightness']:.2f}")
            
            # Generate audio based on the parameters
            generated_audio = generate_ambient_audio(musical_params, duration=3)
            
            # Add to recording buffer
            recording_buffer.append(generated_audio)
            
            # Play the generated audio
            stream_out.write(generated_audio.tobytes())
            
    except KeyboardInterrupt:
        print("\nStopping and saving the recording...")
        
        # Save the entire session
        if recording_buffer:
            mp3_file = save_audio_as_mp3(recording_buffer, RATE)
            print(f"Saved enhanced ambient session to: {mp3_file}")
    finally:
        # Clean up
        stream_in.stop_stream()
        stream_in.close()
        stream_out.stop_stream()
        stream_out.close()
        audio.terminate()

if __name__ == "__main__":
    main()