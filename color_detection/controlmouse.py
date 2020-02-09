import pyautogui 

pyautogui.FAILSAFE = False

xdim, ydim = pyautogui.size()

def execCmd(cmd):
    option = cmd[0]
    if option == -1:
        pass
        # break
    elif option == 0:
        assert len(cmd) == 3
        x = cmd[1]
        y = cmd[2]
        pyautogui.moveTo(xdim-x, y, duration = 0.1)
    elif option == 1:
        x, y = pyautogui.position()
        pyautogui.leftClick(x, y)
    elif option == 2:
        x, y = pyautogui.position()
        pyautogui.rightClick(x, y)
    elif option == 3:
        x = cmd[1]
        pyautogui.scroll(x) 
    elif option == 4:
        x = cmd[1]
        y = cmd[2]
        pyautogui.dragRel(x, y, duration=0.3, button='left')
    else:
        raise Exception("Unimplemented option: {}".format(option))
    # maintain it is still inbounds
    px, py = pyautogui.position()
    if px < 0:
        px = 0
    elif px > xdim:
        px = xdim

    if py < 0:
        py = 0
    elif py > ydim:
        py = ydim

    pyautogui.moveTo(px, py, duration = 0)

    print("position: ", pyautogui.position())

def runtest():

    # xdim, ydim = pyautogui.size()
    # print("xdim: ", xdim)
    # print("ydim: ", ydim)
    pyautogui.moveTo(xdim/2, ydim/2, duration = 0)
    cmd1 = [0, -450, -435]
    cmd2 = [1]
    execCmd(cmd1)
    execCmd(cmd2)

    # while True:
    #     option = int(input("option: "))
    #     exe

# runtest()

