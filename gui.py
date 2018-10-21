import pygame
import string

from utils import *


# A 'Menu' is clickable text in the structured GUI, not the structured GUI itself
class Menu:
    def __init__(self, app, label, top=False, command=None, children=None, disabled=False, hover_msg="", pos=(0, 0)):
        self.app = app # Menus need a reference to the App to access variables like font, font size and padding
        self.label = label # The text displayed on the menu
        self.top = top # Top menus are the uppermost layer of the recursive structure (they have no parent; e.g. File, Edit, Format)
        self.command = command # The function to be called when the menu is clicked
        self.children = children # When hovered over, the menu drops down and reveals its child menus
        self.disabled = disabled # A disabled menu is greyed out and impossible to hover over
        self.hover_msg = hover_msg # A popup displayed to explain to the user when hovered over
        self.x, self.y = pos # While these can be provided, in this application, they are always calculated later

        self.hovered = False # Mouse is inside the Menu's bounding box
        self.held = False # Mouse is hovering and held down

        # Get height, width, and rendered surface
        self.h = app.menu_height
        self.surf = app.font.render(label, 1, (0, 0, 0))
        self.grey_surf = app.font.render(label, 1, (128, 128, 128))
        self.surf_w = self.surf.get_width()
        self.w = self.surf_w + 2*app.padding # There is padding on both the left and the right of the text

        # Render hover_msg
        self.hover_surf = app.font.render(hover_msg, 1, (0, 0, 0))
        self.hover_surf_w = self.hover_surf.get_width()

    def set_pos(self, x, y):
        self.x = x
        self.y = y

    def has_command(self):
        return self.command is not None

    def has_children(self):
        return self.children is not None

    def get_hovered(self): # Hovering features will only work if the menu is not disabled
        return self.hovered and not self.disabled

    def get_held(self):
        return self.held and not self.disabled

    def directly_hovered(self):
        return point_in_rect((self.app.inp.mouse_x, self.app.inp.mouse_y), (self.x, self.y, self.w, self.h))

    def init_children(self): # This method must be called before the menu is functional
        """
            Recurse down the structure, calculating the bounding boxes of each child item
            (excluding height, since this is always 'app.menu_height'
        """
        max_child_w = max(self.children, key=lambda child: child.w).w # The width of the child menu item with the greatest width
        for i, child in enumerate(self.children):
            if self.top:
                child.x = self.x # All children appear directly below the menu item
                child.y = self.y + (i+1)*self.app.menu_height
                child.w = max(self.w, max_child_w)
            else:
                child.x = self.x + self.w
                child.y = self.y + i*self.app.menu_height # The highest child appears directly to the right of the menu item
                child.w = max_child_w # Set the width of all children to the width of the widest child
            if child.has_children():
                child.init_children() # Recurse down the structure

    def tick(self):             # The 'tick' method exists just for clarity; the App could call 'calculate_hover' directly,
        if not self.disabled:   # but that returns things which are unnecessary, possibly causing confusion
            self.calculate_hover()

    def calculate_hover(self):
        """
            There are two ways for a menu to be hovered:
                 1. The mouse is directly inside the menu's bounding box
                 2. The mouse is directly inside the bounding box of one of the menu's children (or grandchildren, etc.)
            Thus, this method is recursive.
        """

        if self.directly_hovered():
            self.hovered = True # This menu item is directly hovered by the mouse (it's in the item's bounding box)
            self.held = self.app.inp.mouse[0] # A menu can only be held if it is at least hovered
            if not self.disabled and self.app.inp.mouse_r[0] and self.has_command(): # Was this box clicked on
                self.command()
                self.hovered = False # Close the menu after selecting an option

            # Set all children of this box to 'neither hovered nor held'
            if self.has_children():
                for child in self.children:
                    child.hovered = False
                    child.held = False
        elif self.get_hovered() and self.has_children(): # self.get_hovered() is a condition so that only non-disabled menus which were open last frame are checked
            # This menu item has children, so it may still satisfy the second menu condition
            self.held = False # It cannot be held if the mouse is not inside the bounding box
            found_one = False # This is true when one of the children is found to be hovered
            for child in self.children:
                if found_one: # If one child is hovered, set all other children to 'not hovered' and don't search their children (i.e. end recursion)
                    child.hovered = False
                    child.held = False
                elif child.calculate_hover(): # Recursively search the child's children to look for one which is moused over
                    found_one = True

            self.hovered = found_one # If one of the children is hovered, then this is hovered (so it can display the hovered child)
        else:
            # The menu item satisfies neither of the two conditions, thus it is neither hovered nor held
            self.hovered = False
            self.held = False

        return self.hovered

    def render(self):
        # Make the menu darker the further down the structure it is, and the closer to held it is
        if not self.top and self.get_held():
            bg = 160
        elif (self.top and self.get_held()) or (not self.top and self.get_hovered()):
            bg = 192
        elif (self.top and self.get_hovered()) or not self.top:
            bg = 224
        else:
            bg = 255
        self.app.screen.fill((bg, bg, bg), rect=(self.x, self.y, self.w, self.h)) # Colour the bounding box

        # Display the rendered text
        if self.top:
            offset = (self.w - self.surf_w) // 2
        else:
            offset = self.app.padding
        if self.disabled:
            self.app.screen.blit(self.grey_surf, (self.x + offset, self.y + self.app.padding))
        else:
            self.app.screen.blit(self.surf, (self.x + offset, self.y + self.app.padding))

        # Render all children, only if this menu is hovered
        if self.has_children() and self.get_hovered():
            for child in self.children:
                child.render()

        # Render hover message if disabled
        if self.hover_msg and self.disabled and self.directly_hovered():
            self.app.screen.fill((224, 224, 224), rect=(
                self.app.inp.mouse_x,
                self.app.inp.mouse_y,
                self.hover_surf_w + 2*self.app.padding,
                self.app.menu_height
            ))
            self.app.screen.blit(self.hover_surf, (
                self.app.inp.mouse_x + self.app.padding,
                self.app.inp.mouse_y + self.app.padding
            ))


class Entry:
    KEY_PRINTABLE = " ',-./0123456789;=[\\]`" + string.ascii_lowercase      # These are the keys that Entries will accept
    KEY_UPPERCASE =" \"<_>?)!@#$%^&*(:+{|}~" + string.ascii_uppercase       # str.upper only affects letters, not symbols
                # The strings need to be lined up because on the bottom one, backslash is used to escape a character

    def __init__(self, app, label, limit, initial_value="", num_mode=False, width=-1):
        self.app = app
        self.label = label
        self.value = str(initial_value)
        self.limit = limit
        self.num_mode = num_mode

        if width == -1:
            self.width = self.app.width // 4
        else:
            self.width = width

    def get(self):
        if self.num_mode:
            if not self.value:
                return 0
            return int(self.value)
        return self.value

    def tick(self):
        if (self.app.inp.keys_p[pygame.K_0] or self.app.inp.keys_p[pygame.K_KP0]) and self.value:
            self.value += "0"               # This needs a special case to prevent numbers like "0000",
                                            # A zero can only be entered if there is a non-zero digit before it
        for i in range(1, 10):
            if self.app.inp.keys_p[pygame.K_0 + i] or self.app.inp.keys_p[pygame.K_KP0 + i]:
                self.value += str(i)

        if self.app.inp.keys_p[pygame.K_BACKSPACE]:
            if self.app.inp.keys[pygame.K_LCTRL] or self.app.inp.keys[pygame.K_RCTRL]:
                self.value = ""
            else:
                self.value = self.value[:-1]

        if self.num_mode:
            if self.value and int(self.value) > self.limit:
                self.value = str(self.limit)
        elif len(self.value) > self.limit:
            self.value = self.value[:self.limit]

        if self.app.inp.keys_p[pygame.K_RETURN]:
            self.active = False
            self.app.pause = False

    def render(self):
        HEIGHT = 2*self.app.font_size + 5*self.app.padding
        LEFT = (self.app.width - self.width) // 2
        TOP = (self.app.height - HEIGHT) // 2
        ENTRY_LEFT = LEFT + self.app.padding
        ENTRY_TOP = (self.app.height - self.app.padding)//2
        self.app.screen.fill((192, 192, 192), rect=(
            LEFT,
            TOP,
            self.width,
            HEIGHT
        ))
        self.app.screen.fill((255, 255, 255), rect=(
            ENTRY_LEFT,
            ENTRY_TOP,
            self.width - 2*self.app.padding,
            self.app.menu_height
        ))
        if self.value:
            value_text = self.value
        else:
            value_text = "0"
        self.app.draw_text(
            (0, 0, 0),
            (ENTRY_LEFT + self.app.padding, ENTRY_TOP + self.app.padding),
            value_text
        )
        self.app.draw_text(
            (0, 0, 0),
            (LEFT + self.app.padding, TOP + self.app.padding),
            self.label
        )
