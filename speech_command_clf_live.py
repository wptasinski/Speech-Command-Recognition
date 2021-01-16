import winsound
import sounddevice as sd
from scipy.io.wavfile import write
import keyboard
import numpy as np
from SoundCommandClf import SoundCommandClf
from os import system

SAMPLE_RATE = 22050  # Sample rate
SECONDS = 1 # Duration of recording
CHANNELS = 1 # Duration of recording

class CommandRecorder(object):
    def __init__(self, sampleRate=SAMPLE_RATE, channels=CHANNELS):
        self.sampleRate = sampleRate
        self.channels = channels

    def record_sound(self, recDuration=1):
        """
        :param recDuration: Duration of recording in seconds
        :return:
        """
        samplesToRecord = int(recDuration * self.sampleRate)
        print('Recording sound!')
        # winsound.Beep(1000, 500)  # Beep at 1000 Hz for 100 ms
        rec = sd.rec(samplesToRecord, samplerate=self.sampleRate, channels=self.channels)
        sd.wait()
        return rec[:,0]

def key_handler():
    try:  # if user pressed other than the given key error will not be shown
        if keyboard.is_pressed('r'):
            system('cls') 
            rec = cr.record_sound()
            print('Finished recording')
            prediction = clf.classify_rec(rec)
            print(prediction[0])
            sd.play(rec, 22050)
        elif keyboard.is_pressed('q'):
            break
    except:
        pass

if __name__ == "__main__":
    clf = SoundCommandClf()
    cr = CommandRecorder()
    while True:
        key_handler()
            # break  # if user pressed a key other than the given key the loop will break

            