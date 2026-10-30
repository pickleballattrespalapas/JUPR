const React = require('react');

// Existing isolated page tests assume the independent review service has passed.
// starting-rating-review.cjs exercises its actual pending/confirm/error behavior.
module.exports = function ReviewedStartingRatings({ onReady, playerIds, matchFormat }) {
  const callback = React.useRef(onReady);
  callback.current = onReady;
  const ids = JSON.stringify(playerIds);
  React.useEffect(() => { callback.current(true, []); }, [ids, matchFormat]);
  return null;
};
