print(978 + 787)
print(978 * 787)

ellie = 978

print(ellie * ellie * ellie * ellie)

name = input("what is your name? ")

print('hello ', name)

sisters_name = input("what is your sisters name? ")

older = input(f"are you older than {sisters_name}? ")

if older == "yes":
  are_you_nice = input(f"are you nice to younger sister, {sisters_name}? ")
  if are_you_nice == "yes":
    print("that's good.")
  if are_you_nice == "no":
    print("you are mean")

if older == "no":
  print("they are annoying")