from multiprocessing import Process, Queue

def square_numbers(numbers, result_queue):
    squared_numbers = [x ** 2 for x in numbers]
    result_queue.put(squared_numbers)

if __name__ == "__main__":
    import time
    
    current_time = time.time()
    data = [1, 2, 3, 4, 5]
    result_queue = Queue()

    num_of_processes = 3
    processes = [Process(target=square_numbers, args=(data, result_queue)) for _ in range(num_of_processes)]

    for process in processes:
        process.start()

    for process in processes:
        process.join()


    results = [result_queue.get() for _ in processes]

    print("Results:", results)
    end_time = time.time()
    print(f"Total time taken: {end_time - current_time:.2f} seconds")