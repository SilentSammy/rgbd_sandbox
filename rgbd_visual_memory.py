"""Position association plus conservative recovery from trusted RGB templates."""

from dataclasses import dataclass

import numpy as np

from rgbd_appearance import compare
from rgbd_object_persistence import ObjectMemory, Track


@dataclass
class VisualTrack(Track):
    templates: tuple = ()
    extent: tuple = ()
    association: str = 'new'
    appearance_verified_at: float = -1.
    last_template_at: float = 0.
    recoveries: int = 0
    label: str | None = None
    label_quality: tuple = ()
    label_conflict: bool = False


def size_compatible(old, new):
    if not old or not new:
        return True
    ratio = np.array(new)/np.maximum(old, .005)
    return bool(.25 <= ratio[0] <= 3.5 and .4 <= ratio[1] <= 2.2 and .55 <= ratio[2] <= 1.7)


class AppearanceMemory(ObjectMemory):
    """Hungarian assignment; uncertain recovery never merges product instances.

    Missing confirmed tracks keep their signatures and OCR identity. Far/gap
    recovery needs strong appearance and mutual ambiguity margins. A hand/box
    merge can remain unassigned instead of corrupting an established track.
    """

    def __init__(self, *, recovery_distance=.8, recovery_margin=.08, **kwargs):
        kwargs.setdefault('forget_after', 30.)
        super().__init__(**kwargs)
        if recovery_distance <= 0 or not 0 <= recovery_margin <= 1:
            raise ValueError('Require positive recovery distance and a margin in [0, 1]')
        self.recovery_distance, self.recovery_margin = recovery_distance, recovery_margin
        self.events, self.blocked, self.matches = [], np.zeros(0, bool), {}

    def clear(self):
        super().clear()
        self.events, self.matches = [], {}

    def update(self, positions, dt, *, observations=None):
        self.events, self.matches = [], {}
        self.blocked = np.zeros(len(positions), bool)
        return super().update(positions, dt, observations=observations)

    def _costs(self, positions, observations, previous_visible):
        if observations is None:
            return super()._costs(positions, observations, previous_visible)
        distance = np.linalg.norm(np.array([t.position for t in self.tracks])[:, None]-positions[None, :], axis=2)
        scores = np.zeros(distance.shape)
        details = {}
        for row, track in enumerate(self.tracks):
            for col, observation in enumerate(observations):
                if observation is None or observation.signature is None or not track.templates or distance[row, col] > self.recovery_distance:
                    continue
                best = None
                for template in track.templates:
                    value = compare(template, observation.signature)
                    if not value.reliable and value.inliers >= 6:
                        # A partial view may be a better query than a full
                        # template containing many currently hidden features.
                        reverse = compare(observation.signature, template)
                        if reverse.score > value.score:
                            value = reverse
                    if best is None or value.score > best.score:
                        best = value
                    if best.score > .98 and best.reliable:
                        break
                scores[row, col], details[row, col] = best.score, best
                if track.confirmed and not size_compatible(track.extent, observation.extent):
                    bigger = observation.extent[2] > 1.7*track.extent[2] or observation.extent[1] > 2.2*track.extent[1]
                    if bigger and (best.reliable or (distance[row, col] <= .03 and best.color >= .5)):
                        self.blocked[col] = True
        for col in np.flatnonzero(self.blocked):
            observation = observations[col]
            # A tall existing package beside a short one is not a merge.
            # Let a compatible, plausible owner protect its own detection.
            for row, track in enumerate(self.tracks):
                value = details.get((row, col))
                if (value is not None and size_compatible(track.extent, observation.extent)
                        and ((distance[row, col] <= .04 and value.score >= .45)
                             or (value.reliable and value.score >= .9))):
                    self.blocked[col] = False
                    break
        costs = np.full(distance.shape, np.inf)
        for row, track in enumerate(self.tracks):
            for col, observation in enumerate(observations):
                if self.blocked[col]:
                    continue
                value = details.get((row, col))
                compatible = observation is None or size_compatible(track.extent, observation.extent)
                if not compatible:
                    continue
                d = distance[row, col]
                other_tracks = np.delete(scores[:, col], row)
                other_detections = np.delete(scores[row], col)
                lead = min(scores[row, col]-max(other_tracks, default=0),
                           scores[row, col]-max(other_detections, default=0))
                reliable = value is not None and value.reliable and value.score >= .9
                feature_count = lambda signature: 0 if signature.descriptors is None else len(signature.descriptors)
                featureless = (observation is not None and observation.signature is not None and track.templates
                               and feature_count(observation.signature) < 6
                               and max(feature_count(t) for t in track.templates) < 6)
                distinctive_color = value is not None and value.color >= .985 and featureless
                recover = (track.confirmed and d <= self.recovery_distance and lead >= self.recovery_margin
                           and (reliable or distinctive_color))
                near = d <= self.match_distance
                if value is not None:
                    competitor = max(other_tracks, default=0)
                    # A clear feature match to another identity takes priority
                    # over being nearest to this one's last position.
                    claimed_elsewhere = competitor >= .9 and competitor-value.score >= self.recovery_margin
                    # An occluded/newly exposed face can have weak features.
                    # Preserve ordinary positional continuity; only a large
                    # jump requires the stricter appearance recovery test.
                    near &= (value.score >= .45 or (d <= .03 and value.color >= .3)) and not claimed_elsewhere
                elif track.templates:
                    near &= previous_visible[track.id] and d <= .03
                if near:
                    score = 0. if value is None else value.score
                    cost = .7*d/self.match_distance+.3*(1-score)
                elif recover:
                    cost = .05+.25*(1-value.score)+.05*d/self.recovery_distance
                else:
                    continue
                costs[row, col] = cost*self.match_distance
                self.matches[track.id, col] = (value, recover and (not previous_visible[track.id] or d > self.match_distance), d)
        if self.blocked.any():
            self.events.append(dict(kind='suspicious_segmentation', detections=np.flatnonzero(self.blocked).tolist()))
        return costs

    def _allow_new(self, index):
        return not self.blocked[index]

    def _make_track(self, position, index, observation):
        return VisualTrack(self.next_id, position.copy(), confirmed=self.probation_period == 0,
            detection_index=index, templates=() if observation is None or observation.signature is None else (observation.signature,),
            extent=() if observation is None else observation.extent, last_template_at=self.elapsed_seconds)

    def _record_match(self, track, index, position, observation, dt):
        value, recovered, distance = self.matches.get((track.id, index), (None, False, 0.))
        super()._record_match(track, index, position, observation, dt)
        track.association = 'appearance' if recovered else ('position+appearance' if value is not None and value.reliable else 'position')
        if value is not None and value.reliable:
            track.appearance_verified_at = self.elapsed_seconds
        if recovered:
            track.recoveries += 1
            self.events.append(dict(kind='appearance_recovery', id=track.id, distance_m=float(distance),
                                    score=value.score, inliers=value.inliers, recoveries=track.recoveries))
        if observation is None or observation.signature is None:
            return
        if not track.templates:
            track.templates, track.extent = (observation.signature,), observation.extent
        elif (track.confirmed and not track.label_conflict and not recovered and distance <= .04
              and value is not None and value.reliable and value.color >= .75
              and self.elapsed_seconds-track.last_template_at >= 1):
            # Keep the initial view; add at most two more trusted views.
            track.templates = (*track.templates, observation.signature) if len(track.templates) < 3 else (track.templates[0], track.templates[-1], observation.signature)
            track.last_template_at = self.elapsed_seconds

    def remember_label(self, identity, best, *, conflict=False):
        for track in self.tracks:
            if track.id == identity:
                track.label_conflict = conflict
                if best is not None:
                    quality = (best['score'], best['ocr_confidence'], best['margin'])
                    if not track.label_quality or quality > track.label_quality:
                        track.label, track.label_quality = best['label'], quality
                return
