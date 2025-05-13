import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
SERVO_PIN = 18
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)

def move_servo(angle):
    duty = angle / 18 + 2
    GPIO.output(SERVO_PIN, True)
    servo.ChangeDutyCycle(duty)
    time.sleep(0.5)
    GPIO.output(SERVO_PIN, False)
    servo.ChangeDutyCycle(0)

def segregate(bean_type):
    if bean_type == "Criollo":
        move_servo(0)
    elif bean_type == "Forastero":
        move_servo(90)
    elif bean_type == "Trinitario":
        move_servo(180)

def cleanup():
    servo.stop()
    GPIO.cleanup()
