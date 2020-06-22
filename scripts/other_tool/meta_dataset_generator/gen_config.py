# coding: utf-8
import json, os


if __name__ == '__main__':
    # config_dir = "D:/Data/Project/Data/MetaData/config/"
    # all_domain = ['schedule', 'navigate', 'weather']
    config_dir = "/Users/lyk/Work/Dialogue/data/DSTC4-TourSg/dstc4_traindev/scripts/config"
    all_domain = ['ITINERARY', 'ACCOMMODATION', 'ATTRACTION', 'FOOD', 'SHOPPING', 'TRANSPORTATION']

    for i in range(len(all_domain)):
        test = all_domain[i]
        dev = all_domain[(i + 1) % len(all_domain)]
        cfg = {'ignore': [], 'dev': dev, 'test': test}
        with open(os.path.join(config_dir, 'config{}.json'.format(i)), 'w') as writer:
            json.dump(cfg, writer)

