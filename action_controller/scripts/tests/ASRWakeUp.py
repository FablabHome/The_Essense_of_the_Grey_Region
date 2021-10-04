import struct
import sys
from os import path

import pvporcupine
import pyaudio
import speech_recognition as sr
from rospkg import RosPack


class WakeUpWord:
    def __init__(self):
        base = RosPack().get_path('rcj_pcms_base') + '/..'
        self.__hey_snippy_model = path.join(base,
                                            'models/PicovoiceWakeUp/hey-snippy__en_linux_2021-10-30-utc_v1_9_0.ppn')
        self.__assistant_model = path.join(base,
                                           'models/PicovoiceWakeUp/assistant__en_linux_2021-10-30-utc_v1_9_0.ppn')
        self.__hey_robie_model = path.join(base,
                                           'models/PicovoiceWakeUp/hey-robie__en_linux_2021-10-30-utc_v1_9_0.ppn')

        self.porcupine = pvporcupine.create(
            library_path=pvporcupine.LIBRARY_PATH,
            model_path=pvporcupine.MODEL_PATH,
            keyword_paths=[
                self.__hey_snippy_model,
                self.__assistant_model,
                self.__hey_robie_model
            ]
        )

        pa = pyaudio.PyAudio()

        self.audio_stream = pa.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length,
        )

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

    def __listen_to_audio(self):
        with self.microphone as source:
            audio = self.recognizer.listen(source)
        return audio

    def __recognize(self, audio):
        try:
            with open('/tmp/speech.wav', 'wb') as wav:
                wav.write(audio.get_wav_data())

            text = self.recognizer.recognize_google(audio, language="en-US")
            return text
        except sr.UnknownValueError:
            print('Speech Recognition could not understand audio')
        except sr.RequestError:
            print('Could not request results from Speech Recognition Service, probably network problems?')

        return ''

    def main(self):
        try:
            while True:
                pcm = self.audio_stream.read(self.porcupine.frame_length)
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                result = self.porcupine.process(pcm)
                if result >= 0:
                    print('Hotword detected, please start speaking')
                    audio = self.__listen_to_audio()
                    text = self.__recognize(audio)
                    print(f'You said: {text}')
        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == '__main__':
    test = WakeUpWord()
    test.main()
