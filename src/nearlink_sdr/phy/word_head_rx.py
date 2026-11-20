"""Actual normalized words and one finite real-CRC head owner;payload remains reference."""

from nearlink_sdr.phy.normalized_word_rx import NormalizedWordPSKReceiver
from nearlink_sdr.phy.strict_rx import _decode_psk_payload
from nearlink_sdr.sim.psk_hard_flow import HeaderFromWordsService

__all__ = ["WordHeadPSKReceiver"]


class WordHeadPSKReceiver(NormalizedWordPSKReceiver):
    """No float head demap/redecode,no alternate CRC answer and no extra word array."""

    def __init__(self, profile, *, head_owner, **options):
        if not isinstance(head_owner, HeaderFromWordsService) or head_owner.mode != "collect":
            raise ValueError("Need this frame's begun finite head owner")
        super().__init__(profile, **options)
        if (
            head_owner.header.profile != self.profile
            or head_owner.start != len(self.reference)
            or head_owner.end != self._head_required
        ):
            raise ValueError("Finite head owner does not match selected profile geometry")
        self.head_owner = head_owner
        self.head_receipt_taken = False

    def _store_normalized_word(self, source_word, output_word, after_samples):
        index = self._active["count"]
        super()._store_normalized_word(source_word, output_word, after_samples)
        if self.head_owner.start <= index < self.head_owner.end and not self.head_owner.offer(
            output_word, index
        ):
            raise RuntimeError("Actual normalization output lost its finite head input credit")

    def accept_header_result(self, *, after_samples):
        """Move the one actual child's receipt;caller cannot supply a success flag."""
        if self.boundary_kind != "header" or self.head_receipt_taken or not self.head_owner.ready:
            raise RuntimeError("Need this active frame's completed finite head")
        self._observe(after_samples)
        row = self.head_owner.take()
        self.head_receipt_taken = True
        self.header_decision_after_filtered_samples = after_samples
        if row["status"] == "header_valid":
            # Only finite field/presence branches;no payload decoding or CRC retry here.
            row["status"] = "truncated_data" if row["data_crc_present"] else "valid_empty_control"
        event = self._accept_selected_header(row)
        if event is not None:
            event["header_decision_after_filtered_samples"] = after_samples
        return event

    def parse_boundary(self, *, after_samples):
        if self.boundary_kind == "header":
            raise RuntimeError("Actual finite head owner must finish on registered edges")
        return super().parse_boundary(after_samples=after_samples)

    def _check_selected_boundary(self):
        a = self._active
        if not self.head_receipt_taken or a["header"] is None or a["count"] != a["end"]:
            raise RuntimeError("Payload continuation needs this owner's valid head and exact end")
        row = _decode_psk_payload(
            self._values[: a["count"]],
            self.profile,
            a["header"],
            carrier_method=self.carrier_method,
            **self._carrier_weight_options,
        )
        return self._event(row)

    def finish(self, *, reason="eof"):
        """EOF/resource exit never executes an unclocked replacement head parser."""
        if reason not in ("eof", "retention_limit", "history_overflow"):
            raise ValueError("Need explicit EOF or finite-resource termination")
        if self.done:
            return []
        if not self._selection_accepted or self._active is None or self._active["count"] == 0:
            return super().finish(reason=reason)
        if self.head_receipt_taken:
            row = dict(self._active["header"], status="truncated_data", data_crc=None, payload=None)
        else:
            count = self._active["count"]
            row = dict(
                status="truncated_sync" if count < len(self.reference) else "truncated_header",
                header_crc=None,
                data_crc=None,
                length_ok=None,
                payload=None,
            )
        if reason != "eof":
            row.update(parser_status=row["status"], status=reason, resource_reason=reason)
        return [self._event(row)]

    def _tag(self, event):
        result = super()._tag(event)
        result.pop("remaining_header_carrier_refinement_LLR_decoder_float", None)
        return result | dict(
            actual_word_hard_head=True,
            head_receipt_taken=self.head_receipt_taken,
            payload_carrier_LLR_decoder_float=True,
        )

    def reference_memory(self):
        result = super().reference_memory()
        result.pop("remaining_header_carrier_refinement_LLR_decoder_float", None)
        return result | dict(
            actual_word_hard_head=True,
            payload_carrier_LLR_decoder_float=True,
            extra_normalized_word_array_bytes=0,
            head_receipt_storage="same child's held receipt;no new head until terminal",
        )
