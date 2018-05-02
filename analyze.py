import numpy as np

data = np.load('gesture_1.npy')

stand_data = data[0:300, :]
crawl_data = data[300:, :]

stand_data = stand_data[~np.all(stand_data == 0, axis=1)]
stand_data
stand_data[:, 4]

crawl_data = crawl_data[~np.all(crawl_data == 0, axis=1)]
crawl_data
crawl_data[:, 2:]

def analyze(mylist, myname):

    print(myname)

    def sub_analyze(sublist, name):
        print(name)
        print('Average: %f' % np.average(sublist))
        print('Standard Deviation: %f' % np.std(sublist))

    aspect_ratios = mylist[:, 4]
    widths = mylist[:, 2]
    heights = mylist[:, 3]

    sub_analyze(aspect_ratios, 'Aspect Ratio')
    sub_analyze(widths, 'Widths')
    sub_analyze(heights, 'Heights')

analyze(stand_data, "STANDING DATA")
analyze(crawl_data, "CRAWLING DATA")
