import sys, os
import threading
import time
import requests
from queue import Queue, Empty
from distutils.dir_util import copy_tree


class Crawler:
    def __init__(self, queue: Queue, targetdir):
        self._queue = queue
        self._targetdir = targetdir

    def crawl(self, target, workers=4):
        stopevent = threading.Event()
        for w in range(workers):
            worker = threading.Thread(target=self.download_from_queue, args=(stopevent,))
            worker.setDaemon(True)
            worker.start()
        while True:
            done = len(os.listdir(self._targetdir))
            left = self._queue.qsize()
            print(f"{self._targetdir}: {done} downloaded, {left} remaining in queue")
            if done >= target or left == 0:
                break
            time.sleep(2)
        stopevent.set()

    def download_from_queue(self, stop):
        while not stop.is_set():
            try:
                nr, url = self._queue.get(timeout=5)
            except Empty:
                break
            image = self.download(url)
            if image:
                name = os.path.join(self._targetdir, f"{nr}.jpg")
                self.save(name, image)
            # time.sleep(0.2)
            self._queue.task_done()

    def download(self, url):
        try:
            r = requests.get(url, timeout=10)
        except:
            return
        if r.headers['Content-Type'] != "image/jpeg":
            return
        return r.content

    def save(self, name, image):
        with open(name, 'wb') as handler:
            handler.write(image)


if __name__ == "__main__":
    urlpath = "urls"
    newpath = "images"
    synsets = os.listdir(urlpath)
    for synset in synsets:
        print("processing", synset)
        fname = os.path.join(urlpath, synset)
        dirname = ".".join(synset.split(".")[:3])
        newdir = os.path.join(newpath, dirname)
        if not os.path.isdir(newdir):
            os.makedirs(newdir)
        urls = Queue()
        counter = 0
        with open(os.path.join(urlpath, synset), encoding="utf-8") as urlfile:
            for url in urlfile:
                if "flickr" in url or "wiki" in url:
                    urls.put((counter, url))
                    counter += 1
        if counter < 100:
            continue
        print(counter, "items")
        crawler = Crawler(urls, newdir)
        crawler.crawl(100)
