import logging
import random
import os
from random import randint
from time import time
from typing import cast

import numpy as np
from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.actions.EndNegotiation import EndNegotiation
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from agents.group69_agent.Group69_Opponent_Model import FrequencyOpponentModelGroup69


def convert_number(value):
    try:
        float(value)
        return value
    except ValueError:
        return None


class Agent69(DefaultParty):
    """
    Template of a Python geniusweb agent.
    """

    def __init__(self):
        super().__init__()
        self.logger: ReportToLogger = self.getReporter()

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: str = None
        self.settings: Settings = None
        self.storage_dir: str = None

        self.sorted_all_bids = None
        self.last_sent_bid: Bid = None
        self.remaining_bids = []
        self.sent_bids = []
        self.last_index = 0

        self.last_received_bid: Bid = None
        self.opponent_model: FrequencyOpponentModelGroup69 = None
        self.logger.log(logging.INFO, "party is initialized")

        self.base_reservation = 0.9
        self.current_reservation = 0.9
        self.modelling_time = 0.6
        self.updated = False

    def notifyChange(self, data: Inform):
        """MUST BE IMPLEMENTED
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be sent to your
        # agent containing all the information about the negotiation session.
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self.progress = self.settings.getProgress()

            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")

            # the profile contains the preferences of the agent over the domain
            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()
            self.stock_bids()
            self.get_bids_within_range(0.9)
            self.opponent_model = FrequencyOpponentModelGroup69.create()
            self.opponent_model = self.opponent_model.With(self.domain, None)
            profile_connection.close()

        # ActionDone informs you of an action (an offer or an accept)
        # that is performed by one of the agents (including yourself).
        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()
            if isinstance(action, Accept):
                print(f"We accepted at utility: {self.profile.getUtility(self.last_sent_bid)}")
            # ignore action if it is our action
            if actor != self.me:
                # obtain the name of the opponent, cutting of the position ID.
                self.other = str(actor).rsplit("_", 1)[0]
                if not self.updated:
                    file_path = os.path.join(str(self.storage_dir), f"{self.other}_data_{self.domain.getName()}.json")
                    # if we want to learn across negotiations for testing purposes can be commented out
                    # self.opponent_model.read_data(file_path)
                    self.updated = True
                # process action done by opponent
                self.opponent_action(action)
        # YourTurn notifies you that it is your turn to act
        elif isinstance(data, YourTurn):
            # execute a turn
            self.my_turn()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(data, Finished):
            self.save_data()
            # terminate the agent MUST BE CALLED
            self.logger.log(logging.INFO, "party is terminating:")
            super().terminate()
        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))

    def getCapabilities(self) -> Capabilities:
        """MUST BE IMPLEMENTED
        Method to indicate to the protocol what the capabilities of this agent are.
        Leave it as is for the ANL 2022 competition

        Returns:
            Capabilities: Capabilities representation class
        """
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    def send_action(self, action: Action):
        """Sends an action to the opponent(s)

        Args:
            action (Action): action of this agent
        """
        self.getConnection().send(action)

    # give a description of your agent
    def getDescription(self) -> str:
        """MUST BE IMPLEMENTED
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "Group 69 agent with following properties"

    def opponent_action(self, action):
        """Process an action that was received from the opponent.

        Args:
            action (Action): action of opponent
        """
        # if it is an offer, set the last received bid
        if isinstance(action, Offer):
            # create opponent model if it was not yet initialised
            if self.opponent_model is None:
                self.opponent_model = FrequencyOpponentModelGroup69.create()

            bid = cast(Offer, action).getBid()

            # update opponent model with bid
            self.opponent_model = self.opponent_model.WithAction(action, self.progress)
            # set bid as last received
            self.last_received_bid = bid

    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """

        # check if the last received offer is good enough
        if self.accept_condition(self.last_received_bid):
            # if so, accept the offer
            print(f"We accepted at utility: {self.profile.getUtility(self.last_received_bid)}")
            action = Accept(self.me, self.last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            bid = self.find_bid()
            if bid is None:
                action = EndNegotiation(self.me)
                print("Ended negotiation without reaching a consensus")
            else:
                action = Offer(self.me, bid)
            self.opponent_model = self.opponent_model.WithMyAction(action, self.progress)

        # send the action
        self.send_action(action)

    def save_data(self):
        """This method is called after the negotiation is finished. It can be used to store data
        for learning capabilities. Note that no extensive calculations can be done within this method.
        Taking too much time might result in your agent being killed, so use it for storage only.
        """

        data = "Data for learning (see README.md)"

        with open(f"{self.storage_dir}/data.md", "w") as f:
            f.write(data)

        if self.other is None or self.opponent_model is None:
            return

        self.opponent_model.save_data(self.storage_dir, self.other)

    ###########################################################################################
    ################################## Example methods below ##################################
    ###########################################################################################

    def accept_condition(self, bid: Bid) -> bool:
        """
        Acceptance condition for the agent.
        Accept the bid if the utility is above the base reservation value.
        If the negotiation is in the last 40% of the time, accept the bid if the utility is above the
        current reservation value.
        """
        if bid is None:
            return False

        # progress of the negotiation session between 0 and 1 (1 is deadline)
        progress = self.progress.get(time() * 1000)
        crt_utility = self.profile.getUtility(bid)
        # print(f"progress: {progress}")

        if progress <= self.modelling_time:
            return crt_utility >= self.base_reservation

        self.current_reservation = self.get_acceptable_utility()
        return crt_utility >= self.current_reservation

    def get_acceptable_utility(self):
        """
        Calculate the acceptable utility for the agent, based on the progress of the negotiation
        and the opponent's strategy.
        If the opponent is hardballing, decrease by a bigger factor for each tick (concede more quickly)
        If the opponent is not hardballing, decrease by a smaller factor for each tick (concede less quickly)
        """
        progress = self.progress.get(time() * 1000)
        hardball = self.is_opponent_hardball()

        # previous solution: linear decrease from 0.9 to 0.5 or 0.4 (depending if the opponent is hardballing)
        #linear_decrease = (progress-self.modelling_time)/(1-self.modelling_time)

        # current solution: still decrease, but adapt more to the opponent's strategy (hardballing or not)

        if hardball: # If opponent is hardballing, we need to concede more quickly
            return self.current_reservation - 0.0001*0.4 # 0.0001 is the change between ticks (1 ms)
                                                        # assuming we update each tick
        else: # If opponent is not hardballing, we can be more strict
            return self.current_reservation - 0.0001*0.25

    def is_opponent_hardball(self):
        """
        Check if the opponent is hardballing
        return: a boolean that checks that:
            - the difference between the first 15% of the last 100 bids and the last 15% of the last 100 bids is less than 0.1
            - the bids are not generally decreasing (more than 60% of the time)
        if the criteria are met, the opponent is hardballing.
        """
        opponent_last_bids = np.array([ self.opponent_model.getUtility(x) for x in self.opponent_model.all_bids[-100:]])
        decreasing = [(1 if x < 0 else 0) for x in np.diff(opponent_last_bids)]
        is_generally_decreasing = np.count_nonzero(decreasing)> 60
        first_max = np.max(opponent_last_bids[:15])
        last_min = np.min(opponent_last_bids[-15:])
        return first_max - last_min < 0.1 and not is_generally_decreasing

    def find_bid(self) -> Bid:
        domain = self.profile.getDomain()
        progress = self.progress.get(time() * 1000)

        # First bid: send highest utility
        if self.last_sent_bid is None:
            top_bid = max(AllBidsList(domain), key=self.profile.getUtility)
            self.last_sent_bid = top_bid
            self.sent_bids.append(self.last_sent_bid)
            return top_bid

        # Phase 1: Boulware greedy start
        if progress <= 0.15:
            min_utility = 0.9
            while self.sorted_all_bids is None or len(self.sorted_all_bids) == 0:
                self.get_bids_within_range(min_utility - 0.05)
                min_utility -= 0.05

            bid = random.choice(self.sorted_all_bids)
            self.sorted_all_bids.remove(bid)
            self.last_sent_bid = bid
            self.sent_bids.append(bid)
            return bid


        # Phase 2: Use opponent model and search best bid
        elif progress <= 0.75:
            bid = self.choose_best_bid(progress)
            if bid is None:
                self.stock_bids()
                bid = self.choose_best_bid(progress)

        # Phase 3: Use opponent model and search for a bid with the mean utility
        else:
            our_utility = self.score_bid(self.last_sent_bid)
            their_utility = self.score_bid(self.last_received_bid)
            target = np.mean([our_utility, their_utility])
            bid = self.choose_suitable_bid(target, progress)
            if bid is None:
                self.stock_bids()
                bid = self.choose_suitable_bid(target, progress)
        if bid is None:
            return bid
        self.remaining_bids.remove(bid)
        self.last_sent_bid = bid
        self.sent_bids.append(self.last_sent_bid)
        return bid

    def stock_bids(self):
        self.remaining_bids = []
        self.sent_bids = []
        all_bids = AllBidsList(self.domain)
        for i in range(all_bids.size()):
            self.remaining_bids.append(all_bids.get(i))

    def score_bid(self, bid: Bid, alpha: float = 0.95, eps: float = 0.1) -> float:
        progress = self.progress.get(time() * 1000)
        our_util = float(self.profile.getUtility(bid))
        time_pressure = 1.0 - progress ** (1 / eps)
        score = alpha * time_pressure * our_util

        if self.opponent_model is not None:
            opp_util = float(self.opponent_model.getUtility(bid))
            score += (1.0 - alpha * time_pressure) * opp_util

        return score

    # def place_bids_in_bins(self):
    #     all_bids = AllBidsList(self.domain)
    #     self.sorted_all_bids = []
    #     for i in range(20):
    #         self.sorted_all_bids.append([])
    #
    #     for j in range(all_bids.size()):
    #         bid = all_bids.get(j)
    #         utility = self.score_bid(bid)
    #         index = int(utility * 20)
    #         if index == 20:
    #             index = 19
    #         self.sorted_all_bids[index].append(bid)

    def get_bids_within_range(self, low: float):
        all_bids = AllBidsList(self.domain)
        self.sorted_all_bids = []
        for i in range(all_bids.size()):
            cur_bid = all_bids.get(i)
            cur_util = self.profile.getUtility(cur_bid)
            if cur_util >= low and self.profile.getUtility(cur_bid) > 0.7:
                self.sorted_all_bids.append(cur_bid)

    def choose_best_bid(self, progress) -> Bid:
        rem_bids = self.remaining_bids
        bid = None
        util = 0
        for cur_bid in rem_bids:
            cur_util = self.score_bid(cur_bid)
            if util < cur_util and self.profile.getUtility(cur_bid) > 0.7:
                util = cur_util
                bid = cur_bid
        return bid

    def choose_suitable_bid(self, target_util: float, progress) -> Bid:
        rem_bids = self.remaining_bids
        bid = None
        max_util = 0
        util = np.inf
        for cur_bid in rem_bids:
            cur_util = self.score_bid(cur_bid)
            if util > abs(target_util - cur_util) and self.profile.getUtility(cur_bid) > 0.7 and not (
            self.sent_bids.__contains__(cur_bid)):
                util = cur_util
                bid = cur_bid
            if max_util < cur_util: max_util = cur_util
        return bid

    # def find_bid(self) -> Bid:
    #     # compose a list of all possible bids
    #     domain = self.profile.getDomain()
    #
    #     progress = self.progress.get(time() * 1000)
    #     ls = self.last_sent_bid
    #     lr = self.last_received_bid
    #
    #     # Sort weights cause for some fucked up reason can't do that in the constructor
    #     if self.sorted_weights is None: self.sort_weights()
    #
    #     # Return highest possible bid
    #     if lr is None:
    #         self.sorted_weights = self.profile.getWeights()
    #         bid = self.sorted_all_bids[0]
    #         self.last_sent_bid = bid
    #         self.sent_bids.append(bid)
    #         return bid[1]
    #
    #     # For the first 15% of the negotiation be greedy and don't go below 0.9 utility
    #     # Choose a random value
    #     min_utility = 0.9
    #     if progress <= 0.15:
    #         while len(self.sorted_all_bids) == 0:
    #             self.sort_bids(min_utility, min_utility - 0.05)
    #             min_utility -= 0.05
    #
    #         # bid = None
    #         # if len(self.sorted_all_bids) > 0:
    #         bid = random.choice(self.sorted_all_bids)
    #         self.sent_bids.append(bid)
    #         self.sorted_all_bids.remove(bid)
    #         self.last_sent_bid = bid
    #         return bid[1]
    #
    #     # Hope opponent model is good enough by now
    #     # If we are within the 75% return the highest combined util of the remaining bid
    #     if progress <= 0.75:
    #         bid = self.choose_best_bid()
    #         self.sent_bids.append(bid)
    #         self.last_sent_bid = bid
    #         return bid
    #
    #     # Search for a bid with the mean utility
    #     our_utility = self.score_bid(ls)
    #     their_utility = self.score_bid(lr)
    #     bid = self.choose_suitable_bid(np.mean([our_utility, their_utility]))
    #     self.sent_bids.append(bid)
    #     self.last_sent_bid = bid
    #     return bid
    #
    # def score_bid(self, bid: Bid, alpha: float = 0.95, eps: float = 0.1) -> float:
    #     """Calculate heuristic score for a bid
    #
    #     Args:
    #         bid (Bid): Bid to score
    #         alpha (float, optional): Trade-off factor between self interested and
    #             altruistic behaviour. Defaults to 0.95.
    #         eps (float, optional): Time pressure factor, balances between conceding
    #             and Boulware behaviour over time. Defaults to 0.1.
    #
    #     Returns:
    #         float: score
    #     """
    #     progress = self.progress.get(time() * 1000)
    #
    #     our_utility = float(self.profile.getUtility(bid))
    #
    #     time_pressure = 1.0 - progress ** (1 / eps)
    #     score = alpha * time_pressure * our_utility
    #
    #     if self.opponent_model is not None:
    #         opponent_utility = float(self.opponent_model.getUtility(bid))
    #         opponent_score = (1.0 - alpha * time_pressure) * opponent_utility
    #         score += opponent_score
    #
    #     return score
    #
    # def sort_weights(self):
    #     dic = self.profile.getWeights()
    #     self.sorted_weights = sorted(dic.items(), key=lambda x: x[1])
    #
    # def sort_bids(self, high, low):
    #     all_bids = AllBidsList(self.domain)
    #     self.sorted_all_bids = []
    #     for i in range(0, all_bids.size()):
    #         utility = self.score_bid(all_bids.get(i))
    #         self.sorted_all_bids.append((utility, all_bids.get(i)))
    #     self.sorted_all_bids.sort(key=lambda x: x[0], reverse=True)
    #     self.sorted_all_bids = list(filter(lambda x: high > x[0] >= low, self.sorted_all_bids))
    #
    # def choose_best_bid(self):
    #     all_bids = AllBidsList(self.domain)
    #     bid = None
    #     util = 0
    #     for i in range(0, all_bids.size()):
    #         cur_bid = all_bids.get(i)
    #         cur_util = self.score_bid(cur_bid)
    #         if util < cur_util and cur_util > 0.5 and not (self.sent_bids.__contains__(cur_bid)):
    #             util = cur_util
    #             bid = cur_bid
    #
    #     return bid
    #
    # def choose_suitable_bid(self, target_util):
    #     all_bids = AllBidsList(self.domain)
    #     bid = None
    #     util = np.inf
    #     for i in range(0, all_bids.size()):
    #         cur_bid = all_bids.get(i)
    #         cur_util = self.score_bid(cur_bid)
    #         if util > abs(cur_util - target_util) and cur_util > 0.5 and not (self.sent_bids.__contains__(cur_bid)):
    #             util = cur_util
    #             bid = cur_bid
    #
    #     return bid
