from geniusweb.profile.utilityspace.UtilitySpace import UtilitySpace
from geniusweb.opponentmodel.OpponentModel import OpponentModel
from decimal import Decimal
from geniusweb.issuevalue.Domain import Domain
from geniusweb.issuevalue.Bid import Bid
from typing import Dict, Optional
from geniusweb.issuevalue.Value import Value
from geniusweb.actions.Action import Action
from geniusweb.progress.Progress import Progress
from geniusweb.actions.Offer import Offer
from geniusweb.references.Parameters import Parameters
from geniusweb.utils import val, HASH, toStr
import os
import json


class FrequencyOpponentModelGroup69(UtilitySpace, OpponentModel):
    '''
    implements an {@link OpponentModel} by counting frequencies of bids placed by
    the opponent.
    <p>
    NOTE: {@link NumberValue}s are also treated as 'discrete', so the frequency
    of one value does not influence the frequency of nearby values
    (as you might expect as {@link NumberValueSetUtilities} is only affected by
    the endpoints).
    <p>
    immutable.
    '''

    _DECIMALS = 4 # accuracy of our computations.
    _ALPHA = Decimal(0.8)
    _BETA = Decimal(0.2)

    def __init__(self, domain: Optional[Domain],
                 freqs: Dict[str, Dict[Value, int]], total: int,
                 resBid: Optional[Bid], change_freqs: Dict[str, int],
                 prev_value: Dict[str, Value], agent69_prev_value: Dict[str, Value], issue_weights: Dict[str, Decimal],
                 all_bids: [Bid], total_changes: int
                 ):
        '''
        internal constructor. DO NOT USE, see create. Assumes the freqs keyset is
        equal to the available issues.

        @param domain the domain. Should not be None
        @param freqs  the observed frequencies for all issue values. This map is
                      assumed to be a fresh private-access only copy.
        @param total  the total number of bids contained in the freqs map. This
                      must be equal to the sum of the Integer values in the
                      {@link #bidFrequencies} for each issue (this is not
                      checked).
        @param resBid the reservation bid. Can be null
        '''
        self._domain = domain
        self._bidFrequencies = freqs
        self._totalBids = total
        self._resBid = resBid
        self._totalChanges = total_changes

        self._frequencyChangePerIssue = change_freqs
        self._previousIssuesValue = prev_value
        self._ourPreviousIssuesValues = agent69_prev_value
        self._issueWeights = issue_weights
        self._pastIssues = {}
        self.all_bids = all_bids

    @staticmethod
    def create() -> "FrequencyOpponentModelGroup69":

        return FrequencyOpponentModelGroup69(None, {}, 0, None, {}, {},{}, {}, [], 0)

    # Override
    def With(self, newDomain: Domain, newResBid: Optional[Bid]) -> "FrequencyOpponentModelGroup69":
        if newDomain == None:
            raise ValueError("domain is not initialized")
        # FIXME merge already available frequencies?
        return FrequencyOpponentModelGroup69(newDomain,
                                             {iss: {} for iss in newDomain.getIssues()},
                                             0, newResBid, {iss: 0 for iss in newDomain.getIssues()},
                                             {iss: None for iss in newDomain.getIssues()},
                                             {iss: None for iss in newDomain.getIssues()},
                                             {iss: Decimal(0) for iss in newDomain.getIssues()},
                                             [], total_changes=0)

    # Override
    def getUtility(self, bid: Bid) -> Decimal:
        if self._domain == None:
            raise ValueError("domain is not initialized")
        if self._totalBids == 0:
            return Decimal(1)
        sum = Decimal(0)
        # issues do not have equal weights
        for issue in val(self._domain).getIssues():
            if issue in bid.getIssues():
                sum = sum + self._issueWeights[issue] * self._getFraction(issue, val(bid.getValue(issue)))
        return round(sum / len(self._bidFrequencies), FrequencyOpponentModelGroup69._DECIMALS)

    # Override
    def getName(self) -> str:
        if self._domain == None:
            raise ValueError("domain is not initialized")
        return "FreqOppModel" + str(hash(self)) + "For" + str(self._domain)

    # Override
    def getDomain(self) -> Domain:
        return val(self._domain)

    # Override
    def WithAction(self, action: Action, progress: Progress) -> "FrequencyOpponentModelGroup69":
        if self._domain == None:
            raise ValueError("domain is not initialized")

        if not isinstance(action, Offer):
            return self

        bid: Bid = action.getBid()
        self.all_bids.append(bid)
        newFreqs: Dict[str, Dict[Value, int]] = self.cloneMap(self._bidFrequencies)
        for issue in self._domain.getIssues():  # type:ignore
            freqs: Dict[Value, int] = newFreqs[issue]
            value = bid.getValue(issue)
            if value != None:
                oldfreq = 0
                if value in freqs:
                    oldfreq = freqs[value]
                # calculate value frequency
                freqs[value] = oldfreq + 1

                # update the changes in values
                if self._previousIssuesValue[issue] != value:
                    self._frequencyChangePerIssue[issue] = self._frequencyChangePerIssue[issue] + 1
                    self._previousIssuesValue[issue] = value

        total_weight = 0
        # calculate weights based on issue changes
        for issue in self._issueWeights.keys():
            frequency_weight = Decimal(1) / (Decimal(1) + Decimal(self._frequencyChangePerIssue[issue]))
            counter_response_weight = Decimal(0)
            if (self._ourPreviousIssuesValues[issue] is not None
                    and self._ourPreviousIssuesValues[issue] != bid.getValue(issue)):
                counter_response_weight = Decimal(0.2)

            weight = (FrequencyOpponentModelGroup69._ALPHA * frequency_weight
                      + FrequencyOpponentModelGroup69._BETA * counter_response_weight)
            gamma = Decimal(0.7)
            combined_weight = gamma*self._pastIssues.get(issue, Decimal(0)) + (1-gamma)*weight
            self._issueWeights[issue] = combined_weight
            total_weight += combined_weight
        for issue in self._issueWeights.keys():
            self._issueWeights[issue] = self._issueWeights[issue]/total_weight

        return FrequencyOpponentModelGroup69(self._domain, newFreqs,
                                             self._totalBids + 1, self._resBid, dict(self._frequencyChangePerIssue),
                                             dict(self._previousIssuesValue), dict(self._ourPreviousIssuesValues),
                                             dict(self._issueWeights), list(self.all_bids), self._totalChanges)
    def WithMyAction(self, action: Action, progress: Progress) -> "FrequencyOpponentModelGroup69":
        if self._domain == None:
            raise ValueError("domain is not initialized")

        if not isinstance(action, Offer):
            return self

        bid: Bid = action.getBid()
        ourValues = {}
        for issue in self._domain.getIssues():
            ourValues[issue] = bid.getValue(issue)

        return FrequencyOpponentModelGroup69(self._domain, dict(self._bidFrequencies),
                                             self._totalBids + 1, self._resBid, dict(self._frequencyChangePerIssue),
                                             dict(self._previousIssuesValue), ourValues,
                                             dict(self._issueWeights), list(self.all_bids), self._totalChanges)
    def getCounts(self, issue: str) -> Dict[Value, int]:
        '''
        @param issue the issue to get frequency info for
        @return a map containing a map of values and the number of times that
                value was used in previous bids. Values that are possible but not
                in the map have frequency 0.
        '''
        if self._domain == None:
            raise ValueError("domain is not initialized")
        if not issue in self._bidFrequencies:
            return {}
        return dict(self._bidFrequencies.get(issue))  # type:ignore

    # Override
    def WithParameters(self, parameters: Parameters) -> OpponentModel:
        return self  # ignore parameters

    def _getFraction(self, issue: str, value: Value) -> Decimal:
        '''
        @param issue the issue to check
        @param value the value to check
        @return the fraction of the total cases that bids contained given value
                for the issue.
        '''
        if self._totalBids == 0:
            return Decimal(1)
        if not (issue in self._bidFrequencies and value in self._bidFrequencies[issue]):
            return Decimal(0)
        freq: int = self._bidFrequencies[issue][value]
        return round(Decimal(freq) / self._totalBids, FrequencyOpponentModelGroup69._DECIMALS)  # type:ignore

    @staticmethod
    def cloneMap(freqs: Dict[str, Dict[Value, int]]) -> Dict[str, Dict[Value, int]]:
        '''
        @param freqs
        @return deep copy of freqs map.
        '''
        map: Dict[str, Dict[Value, int]] = {}
        for issue in freqs:
            map[issue] = dict(freqs[issue])
        return map

    # Override
    def getReservationBid(self) -> Optional[Bid]:
        return self._resBid

    def save_data(self, storage_dir, other):
        issue_data = {}
        for issue, weight in self._issueWeights.items():
            value_utilities = {}
            for value, count in self._bidFrequencies.get(issue, {}).items():
                value_utilities[value.getValue()] = self._bidFrequencies[issue][value]

            issue_data[issue] = {
                "DiscreteValueSetUtilities": {
                    "valueUtilities": value_utilities
                },
                "Weight": float(weight)
            }

        # self._resBid = self.all_bids[self.all_bids.index(min([self.getUtility(x) for x in self.all_bids]))]
        changes_sum = 0
        for issue in self._issueWeights:
            changes_sum += sum(self._bidFrequencies[issue].values())


        # Combine data
        data_to_save = {
            "Issues": issue_data,
            "Reservation": self._resBid,
            "Bids Exchanged": self._totalChanges + changes_sum
        }

        # Save to file
        file_path = os.path.join(storage_dir, f"{other}_data.json")
        with open(file_path, "w") as f:
            json.dump(data_to_save, f, indent=4)

    def read_data(self, file_path: str):
        '''
        Reads a JSON file to load past negotiation data and updates issue weights and frequencies.
        '''
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            return

        with open(file_path, 'r') as f:
            data = json.load(f)

        previous_total_bids = data.get('Bids Exchanged', 0)
        self._totalChanges = previous_total_bids
        previous_issue_data = data.get('Issues', {})
        self._resBid = data.get('Reservation', None)

        # Update frequencies and weights
        for issue, issue_info in previous_issue_data.items():
            # Update issue weights
            if 'Weight' in issue_info:
                self._pastIssues[issue] = Decimal(issue_info['Weight'])

            # Update value frequencies
            value_utilities = issue_info.get('DiscreteValueSetUtilities', {}).get('valueUtilities', {})
            for value_str, utility in value_utilities.items():
                value = Value(value_str)
                if issue not in self._bidFrequencies:
                    self._bidFrequencies[issue] = {}

                # Estimate frequency based on utility and previous rounds
                if value in self._bidFrequencies[issue]:
                    self._bidFrequencies[issue][value] += utility
                else:
                    self._bidFrequencies[issue][value] = utility

        print("Data successfully read and integrated from JSON file.")

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self._domain == other._domain and \
            self._bidFrequencies == other._bidFrequencies and \
            self._totalBids == other._totalBids and \
            self._resBid == other._resBid

    def __hash__(self):
        return HASH((self._domain, self._bidFrequencies, self._totalBids, self._resBid))

    # Override

    # Override
    def __repr__(self) -> str:
        return "FrequencyOpponentModel[" + str(self._totalBids) + "," + \
            toStr(self._bidFrequencies) + "]"
