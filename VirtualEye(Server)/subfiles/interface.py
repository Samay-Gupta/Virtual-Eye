import pygame

class Display:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont('Times', 24)
        self.screen = pygame.display.set_mode((800,600))
        pygame.display.set_caption('Virtual Eye Server')
        icon = pygame.image.load('subfiles\\images\\serv_logo.jpg')
        pygame.display.set_icon(icon)
        self.screen.fill((0, 0, 0))
        self.text_history = []
        self.permanent_text = []

    def write(self, text=""):
        print(text)
        if text != "":
            self.screen.fill((0, 0, 0))
            self.text_history.append(text)
            ind = 0
            for y in range(550, -25, -25):
                if ind >= len(self.text_history):
                    break
                txt = self.text_history[-ind-1]
                obj = self.font.render(txt, False, (0, 255, 255))
                self.screen.blit(obj, (10, y))
                ind += 1
        for y in range(len(self.permanent_text)):
            obj = self.font.render(self.permanent_text[y], False, (0, 255, 255))
            self.screen.blit(obj, (500, 25*y))
        pygame.display.update()

    def add_permanent_tag(self, tag_lines):
        self.permanent_text = list(tag_lines[:3])
        self.write("")
