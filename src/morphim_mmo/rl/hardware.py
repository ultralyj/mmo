"""Joint and collision metadata from the pinned Wuji URDF and vendor MJCF."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class JointHardware:
    lower: float
    upper: float
    effort_nm: float
    velocity_rad_s: float
    stiffness_nm_rad: float
    damping_nm_s_rad: float
    armature_kg_m2: float


def load_hand_hardware(urdf_path: str | Path):
    urdf = ET.parse(urdf_path).getroot()
    mjcf = ET.parse(Path(urdf_path).parents[1] / 'mjcf/right.xml').getroot()
    actuators = {x.get('joint'): x for x in mjcf.findall('actuator/position')}
    armatures = {x.get('name'): float(x.get('armature', mjcf.find('default/joint').get('armature')))
                 for x in mjcf.findall('.//worldbody//joint')}
    joints = {}
    adjacent = set()
    for joint in urdf.findall('joint'):
        adjacent.add(frozenset([joint.find('parent').get('link'), joint.find('child').get('link')]))
        limit = joint.find('limit')
        if limit is None:
            continue
        name = joint.get('name'); actuator = actuators[name]
        joints[name] = JointHardware(
            lower=float(limit.get('lower')), upper=float(limit.get('upper')),
            effort_nm=float(limit.get('effort')), velocity_rad_s=float(limit.get('velocity')),
            stiffness_nm_rad=float(actuator.get('kp')), damping_nm_s_rad=float(actuator.get('kv')),
            armature_kg_m2=armatures[name])
    # Fixed sensor/tip frames are one physical rigid body with their parent.
    # Adjacency must refer to rigid bodies, not only URDF frame names.
    clusters = {link.get('name'): {link.get('name')} for link in urdf.findall('link')}
    for joint in urdf.findall('joint'):
        if joint.get('type') == 'fixed':
            left = joint.find('parent').get('link'); right = joint.find('child').get('link')
            merged = clusters[left] | clusters[right]
            for name in merged:
                clusters[name] = merged
    frame_exclusions = adjacent | {frozenset([x.get('body1'), x.get('body2')]) for x in mjcf.findall('contact/exclude')}
    exclusions = set()
    for cluster in clusters.values():
        for left in cluster:
            for right in cluster:
                if left != right:
                    exclusions.add(frozenset([left, right]))
    for pair in frame_exclusions:
        left, right = tuple(pair)
        for first in clusters[left]:
            for second in clusters[right]:
                if first != second:
                    exclusions.add(frozenset([first, second]))
    return joints, exclusions


def collision_ignore_masks(link_names, excluded_pairs):
    """Cover excluded cliques with bits, preserving every nonexcluded pair."""
    names = set(link_names)
    graph = {pair for pair in excluded_pairs if pair <= names}
    remaining = set(graph)
    masks = {name: 0 for name in names}
    bit = 0
    while remaining:
        clique = set(min(tuple(sorted(pair)) for pair in remaining))
        for candidate in sorted(names - clique):
            if all(frozenset([candidate, member]) in graph for member in clique):
                clique.add(candidate)
        if bit >= 32:
            raise ValueError('collision exclusions exceed the 32-bit PhysX ignore mask')
        for name in clique:
            masks[name] |= 1 << bit
        remaining = {pair for pair in remaining if not pair <= clique}
        bit += 1
    return masks
