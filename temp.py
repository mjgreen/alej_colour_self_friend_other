# choose one of 6 combinations of j,k,l mapping to redself; bluefriend; othergreen
#redself   bluefriend  greenother
#1=j              k                    l
#2=j              l                     k
#3=k             j                     l
#4=k             l                     j
#5=l              j                     k
#6=l              k                    j

# create strings from keymap for use in instructions
#e.g., instr_1 = "press" + keymap1key1 + "for redself"




import random
mapping_number = random.randint(1,6)
match mapping_number:
    case 1:
        redself    = "j"
        bluefriend = "k"
        greenother = "l"
    case 2:
        redself    = "j"
        bluefriend = "l"
        greenother = "k"
    case 3:
        redself    = "k"
        bluefriend = "j"
        greenother = "l"
    case 4:
        redself    = "k"
        bluefriend = "l"
        greenother = "j"
    case 5:
        redself    = "l"
        bluefriend = "j"
        greenother = "k"
    case 6:
        redself    = "l"
        bluefriend = "k"
        greenother = "j"                        
    case _:
        print("oops")

instruct_color_condition = "Press " + redself + " for red; " + bluefriend + " for blue; or " + greenother + " for green"
instruct_identity_condition  = "Press " + redself + " for self; " + bluefriend + " for friend; or " + greenother + " for other"


instruct_color_condition = "In the following task, you will need to report the color of the word. \n- If the word is presented in blue, press " + bluefriend + "\n- If the word is presented in red, press " + redself + "\n- If the word is presented in green, press " + greenother + "\n\npress the space key to start"

instruct_identity_condition = "In the following task, you will need to report the identity of the word.\n- If the word is self, press " + redself + "\n- If the word is friend, press " + bluefriend + "\n- If the word is other, press " + greenother + "\n\npress the space key to start"