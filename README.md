import os, sys
import time
import RPi.GPIO as GPIO

PIN_DOOR_OPEN = 12 #input (32 for board)
PIN_DOOR_CLOSE = 13 #input (33 for board)
PIN_LB = 27 #input (13 for board)
PIN_LT = 22 #input (15 for board)
TIMER = 10 #time out

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_DOOR_OPEN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(PIN_DOOR_CLOSE, GPIO.OUT, initial=GPIO.LOW)

GPIO.setup(PIN_LB, GPIO.IN)
GPIO.setup(PIN_LT, GPIO.IN)
DOOR_CLOSE = True

try:
	while True:
		input_c = input("Enter key o to open the door and c to close it")
		if input_c == 'o' and DOOR_CLOSE:
			# set the pin 32 high
			print ('opening the door')
			GPIO.output(PIN_DOOR_OPEN, GPIO.HIGH)
			# start the timer
			st_time = time.time()
			while time.time() - st_time < 10 and GPIO.input(PIN_LB) == GPIO.LOW:
				pass
			# put GPIO 32 LOW now as door has opened
			print ('door opened/timer expires')
			GPIO.output(PIN_DOOR_OPEN, GPIO.LOW)
			DOOR_CLOSE = False
		elif input_c == 'c' and (not DOOR_CLOSE):
			print ('closing the door')
			GPIO.output(PIN_DOOR_CLOSE, GPIO.HIGH)
			st_time = time.time()
			while time.time() -st_time <10 and GPIO.input(PIN_LT) == GPIO.LOW:
				pass

			print ('door is closed/timer expires')
			GPIO.output(PIN_DOOR_CLOSE, GPIO.LOW)
			DOOR_CLOSE = True
		else:
			print ('something went wrong. press ctrl +c and try again')



finally:
	GPIO.cleanup()





