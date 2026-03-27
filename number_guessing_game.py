import random

print("Welcome to the number guessing game. For this game you will determine a low number and a high number for me to choose from.")
print("I will choose a number between the two you provided and it will be up to you to guess the correct one in as little attempts possible.")
print("Good luck!")



"""
    1. Prompt the user with a hello message on start up giving brief instructions on the game.
    2. user input to determine the low number of the guessing range.
    3. user input to determine the high number of the guessing game.
    4. Confirmation that a number has been chosen at random between the two values.
    5. user input to guess.
        a. return too low and a new user input
        b. return too high and a new user input
        c. Correct! print number of guesses. 
    6. Play again?
        a. clear any cached data and provide a new prompt. Maybe clear command prompt? maybe?
        b. end. 

    ensure user input is only a int. (handle error gracefully and return prompt to user)
    initial value not below 1. (handle error gracefully and return prompt to user)
    comma's in input? 
"""


def initGame():
    global low, high, guesses, number
    guesses = 0

    chooseLow()
    chooseHigh()
    number = random.randint(low, high)
    doYouWannaPlayAGame()

    while True:
        again = input("Do you want to play again? (Yes/No): ").strip().lower()
        if again == "yes":
            initGame()
            return
        elif again == "no":
            print("Thanks for playing!")
            return
        else:
            print("Please enter 'Yes' or 'No'")


def chooseLow():
    global low
    while True:
        try:
            low = int(input("Enter the low number in the range:"))
            break
        except ValueError:
            print("That was not a valid number. Please try again...")

def chooseHigh():
    global high
    while True:
        try:
            high = int(input("Enter the high number in the range:"))
            if high <= low:
                print("High number cannot be less than or equal to the low number. Please try again...")
                continue
            break
        except ValueError:
            print("That was not a valid number. Please try again...")


def doYouWannaPlayAGame():
    global guesses, number

    while True:
        try:
            userChoice = int(input("Guess the number: "))
            guesses += 1

            if userChoice > number:
                print("Too High!")
            elif userChoice < number:
                print("Too Low!")
            else:
                print("Correct! Guesses:", guesses)
                return
        except ValueError:
            print("That was not a valid number. Please try again...")

initGame()