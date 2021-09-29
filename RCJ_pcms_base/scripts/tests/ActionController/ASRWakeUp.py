import sys

import speech_recognition as sr


class WakeUpWord:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        self.stop_listening = self.recognizer.listen_in_background(self.microphone, self.callback)
        self.hotword_detected = False

    def callback(self, _, audio):
        text = self.__recognize(audio)
        if 'hey' in text:
            self.hotword_detected = True

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
                if self.hotword_detected:
                    self.stop_listening(wait_for_stop=True)
                    print('Hotword detected, please start speaking')
                    audio = self.__listen_to_audio()
                    text = self.__recognize(audio)
                    print(f'You said: {text}')
                    self.stop_listening = self.recognizer.listen_in_background(self.microphone, self.callback)
                    self.hotword_detected = False
        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == '__main__':
    test = WakeUpWord()
    test.main()
