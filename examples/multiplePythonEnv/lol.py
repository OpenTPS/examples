from multiprocessing import shared_memory

try:
    shm = shared_memory.SharedMemory(name='sharedArray')
    shm.close()
    shm.unlink()
    print("Shared memory 'sharedArray' successfully removed.")
except FileNotFoundError:
    print("No shared memory named 'sharedArray' found.")
