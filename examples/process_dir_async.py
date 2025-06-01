import asyncio
import sys
from asyncio import Queue
from os import scandir
from pathlib import Path

import aiofiles

# Add parent directory to Python path to access custom modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tqdm
from pydantic import BaseModel
from typing import Optional
from ifr_clients import IFRClientAsync
from ifr_clients.common_utils import to_chunks


class RecTask(BaseModel):
    """
    Represents a single recognition task in the processing pipeline.

    Attributes:
        path (str): File path of the image to process
        binary (Optional[bytes]): Binary content of the image file
        result (Optional[dict]): Recognition results from the API
    """
    path: str
    binary: Optional[bytes] = None
    result: Optional[dict] = None


def scantree(path: str, ret_path: bool = True):
    """
    Recursively scans a directory tree and yields file paths or DirEntry objects.

    Args:
        path: Root directory to start scanning
        ret_path: If True, yields file paths; otherwise yields DirEntry objects

    Yields:
        Either file paths (str) or DirEntry objects depending on ret_path
    """
    for entry in scandir(path):
        if entry.is_dir(follow_symlinks=False):
            # Recurse into subdirectories
            yield from scantree(entry.path, ret_path=ret_path)
        else:
            # Yield file entry based on return type preference
            yield entry.path if ret_path else entry


async def file_reader_worker(in_queue: Queue, out_queue: Queue):
    """
    Asynchronous worker that reads file contents from disk.

    Processes batches of RecTasks by reading their file contents and passing
    them to the next stage in the pipeline.

    Args:
        in_queue: Input queue containing batches of RecTasks
        out_queue: Output queue for processed tasks with file content
    """
    while True:
        # Get next batch of tasks from input queue
        tasks = await in_queue.get()
        if tasks is None:  # Termination signal
            break

        # Process each task in the batch
        for task in tasks:
            try:
                # Asynchronously read file content
                async with aiofiles.open(task.path, 'rb') as f:
                    task.binary = await f.read()
            except Exception as e:
                # Handle file read errors
                print(f"Error reading {task.path}: {str(e)}")
                task.binary = None

        # Pass processed batch to next stage
        await out_queue.put(tasks)
        in_queue.task_done()


async def recognition_worker(
        in_queue: Queue,
        out_queue: Queue,
        host: str,
        port: int
):
    """
    Worker that processes images through the face recognition API.

    Takes batches of RecTasks with file content, sends them to the recognition API,
    and attaches results to the tasks.

    Args:
        in_queue: Input queue with tasks containing file content
        out_queue: Output queue for tasks with recognition results
        host: Recognition API host URL
        port: Recognition API port
    """
    # Initialize recognition client
    client = IFRClientAsync(host=host, port=port)
    await client.start()

    try:
        while True:
            # Get next batch of tasks
            tasks = await in_queue.get()
            if tasks is None:  # Termination signal
                break

            # Filter out tasks that failed file reading
            valid_tasks = [t for t in tasks if t.binary is not None]
            images = [t.binary for t in valid_tasks]

            try:
                # Process batch through recognition API
                resp = await client.extract(
                    data=images,
                    mode='data',
                    raw_response=True,  # Return raw dict instead of Pydantic model
                    use_msgpack=True  # Use efficient binary serialization
                )

                # Attach results to corresponding tasks
                for i, item in enumerate(resp['data']):
                    valid_tasks[i].result = item
            except Exception as e:
                # Handle API processing errors
                print(f"Recognition error: {str(e)}")
                for task in valid_tasks:
                    # Create error result for failed tasks
                    task.result = {'status': 'error', 'message': str(e)}

            # Pass results to next stage
            await out_queue.put(tasks)
            in_queue.task_done()
    finally:
        # Ensure client session is closed properly
        await client.close()


async def results_processing(in_queue: Queue):
    """
    Handles recognition results and displays progress.

    Processes completed recognition tasks, updates progress bar,
    and displays processing statistics.

    You can modify this method i.e. for writing results to database.

    Args:
        in_queue: Input queue containing completed RecTasks with results
    """
    processed_count = 0
    # Initialize progress bar
    with tqdm.tqdm() as pbar:
        while True:
            # Get next batch of completed tasks
            tasks = await in_queue.get()
            if tasks is None:  # Termination signal
                break

            # Process each task in the batch
            for task in tasks:
                processed_count += 1
                if task.result:
                    # Extract and display result info
                    status = task.result.get('status', 'unknown')
                    faces = task.result.get('faces', [])
                    pbar.write(
                        f"Processed: {task.path} | Status: {status} | Faces: {len(faces)}"
                    )
                else:
                    # Handle tasks without results
                    print(f"Failed: {task.path} - No result")
                # Update progress bar
                pbar.update(1)
            in_queue.task_done()

    # Display final statistics
    print(f"\nTotal processed files: {processed_count}")


async def main(images_dir: str, host: str, port: int):
    """
    Main pipeline orchestrator.

    Sets up processing queues, worker pools, and manages the workflow:
    1. File reading → 2. Recognition processing → 3. Results handling

    Args:
        images_dir: Directory containing images to process
        host: Recognition API host URL
        port: Recognition API port
    """
    # Configuration parameters
    concurrency = 10  # Number of parallel workers per stage
    file_read_queue = Queue(maxsize=concurrency)
    recognition_queue = Queue(maxsize=concurrency)
    results_queue = Queue(maxsize=concurrency)

    # Worker pools
    read_workers = []
    rec_workers = []

    # Create file reader workers
    for _ in range(concurrency):
        read_workers.append(asyncio.create_task(
            file_reader_worker(file_read_queue, recognition_queue)
        ))

    # Create recognition workers
    for _ in range(concurrency):
        rec_workers.append(asyncio.create_task(
            recognition_worker(recognition_queue, results_queue, host, port)
        ))

    # Create results processor
    results_worker = asyncio.create_task(results_processing(results_queue))

    # Collect image paths recursively
    images = scantree(images_dir)
    # Group images into batches of 10
    image_chunks = to_chunks(images, size=10)

    # Submit batches to processing pipeline
    for image_chunk in image_chunks:
        # Create tasks for each image in batch
        rec_tasks = [RecTask(path=path) for path in image_chunk]
        await file_read_queue.put(rec_tasks)

    # Wait for all queues to empty
    for q in (file_read_queue, recognition_queue, results_queue):
        await q.join()

    # Send termination signals to workers
    for _ in range(concurrency):
        await file_read_queue.put(None)
        await recognition_queue.put(None)
    await results_queue.put(None)

    # Cancel and wait for workers to finish
    all_workers = read_workers + rec_workers + [results_worker]
    for task in all_workers:
        task.cancel()
    # Gracefully handle cancellations
    await asyncio.gather(*all_workers, return_exceptions=True)


if __name__ == "__main__":
    # Configuration
    host = 'http://localhost'  # Recognition API host
    port = 18081  # Recognition API port
    images_dir = 'misc/test_images'  # Directory with images to process

    # Run main pipeline
    asyncio.run(main(
        images_dir=images_dir,
        host=host,
        port=port
    ))
