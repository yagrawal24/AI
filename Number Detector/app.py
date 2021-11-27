import pygame
import sys
import cv2
from pygame.locals import *
from intro_pytorch import *

train_loader = get_data_loader()
test_loader = get_data_loader(False)
model = build_model()
criterion = nn.CrossEntropyLoss()
train_model(model, train_loader, criterion, T=5)
evaluate_model(model, test_loader, criterion, show_loss = True)

WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDRYINC = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
IMAGESAVE = False
LABELS = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}

pygame.init()
FONT = pygame.font.Font("freesansbold.ttf", 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
DISPLAYSURF.fill(BLACK)
WHILE_INT = DISPLAYSURF.map_rgb(WHITE)
pygame.display.set_caption("Number Identifior")

iswriting = False
number_xcord = []
number_ycord = []
image_cnt = 1
PREDICT = True

while True:

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            number_ycord.append(ycord)
            number_xcord.append(xcord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)
            rect_min_x, rect_max_x = max(number_xcord[0], 0), min(WINDOWSIZEX, number_xcord[-1])
            rect_min_y, rect_max_y = max(number_ycord[0], 0), min(number_ycord[-1], WINDOWSIZEY)
            number_xcord = []
            number_ycord = []
            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y: rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png")
                image_cnt += 1
            if PREDICT:

                image = cv2.resize(img_arr, (28,28))
                image = torch.tensor([image])
                print(image)

                pred = model(image)
                pred = pred[0]
                pred = pred.tolist()
                print(pred)
                label = str(LABELS[np.argmax(pred)])

                textSurface = FONT.render(label, True, RED, WHITE)
                textRectObj = textSurface.get_rect()
                textRectObj.left, textRectObj.bottom = rect_min_x, rect_max_y

                DISPLAYSURF.blit(textSurface, textRectObj)
        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)
    pygame.display.update()