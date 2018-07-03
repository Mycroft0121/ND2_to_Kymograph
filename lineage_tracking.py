# implement linked list in python


class Cell(object):
    def __init__(self, frame_ID, birth_time, length, intensities, prev=None, next=None,  mother=None, daughter=None, lost_time=None, pole_tracker=None):
        self.frame_ID     = frame_ID     # (t, #), use to retrieve cell properties
        # self.prev         = prev         # the same cell in the previous frame, recorded as Cell
        # self.next         = next         # the same cell in the next frame, recorded as Cell
        # self.birth_time   = birth_time   # t, when was this cell born
        # self.mother       = mother       # frame ID of its mother
        # self.daughter     = daughter     # frame ID of its daughter
        self.lost_time    = lost_time    # t, when the cell was lost
        self.pole_tracker = pole_tracker # (i,j), pole tracker to study ageing effect
        self.length       = length       # cell length
        self.intensities  = intensities  # reporters intensity list

    # def give_birth(self, next, birth_time):
    #     self.next = next
    #     # update the pole_tracker
    #     self.pole_tracker[0] += 1

    def get_lost(self, lost_time):
        self.lost_time = lost_time


class Lineage(object):
    def __init__(self, birth_time, mother=None):
        self.head = None
        self.tail = None
        self.birth_time = birth_time
        self.size = 0
        self.mother = mother
        self.daughters = None    # will be a list of lineage objects

    def append(self, frame_ID, birth_time):
        if self.head is None:
            new_cell = Cell(frame_ID,birth_time)
            self.head = new_cell
        else:
            new_cell = Cell(frame_ID, birth_time, prev=self.tail)
            self.tail = new_cell
            self.tail.next = new_cell
        self.size += 1

    def remove(self, frame_ID):
        cur_cell = self.head
        while cur_cell is not None:
            if cur_cell.frame_ID == frame_ID:
                if cur_cell.prev is not None:
                    cur_cell.prev.next = cur_cell.next
                    cur_cell.next.prev = cur_cell.prev
                else:
                    self.head = cur_cell.next
                    cur_cell.next.prev = None
                self.size -= 1
                return
            cur_cell = cur_cell.next
        print('removal error')
        return 1

    def give_birth(self, frame_ID):
        return


# algorithms to identify lineage
# input data structure: frame_IDs for a single trench
#
# 1. starting from the mother, create a lineage, track it along time
# 2. once a division event happened, initiate a new lineage with the daughter
# 3. repeat the tracking backwards from the youngest daughter to the oldest





