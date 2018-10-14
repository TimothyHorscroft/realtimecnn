import pygame
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

        if point_in_rect((self.app.inp.mouse_x, self.app.inp.mouse_y), (self.x, self.y, self.w, self.h)):
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
