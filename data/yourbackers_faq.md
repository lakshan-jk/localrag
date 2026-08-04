# YourBackers — Frequently Asked Questions

## How is video content protected from piracy?
Premium content uses Multi-DRM (Pallycon Widevine). On top of DRM, media lives in a private
Amazon S3 bucket and is only served through CloudFront using signed URLs, so links cannot be
shared or accessed without authorization.

## Why does playback stay smooth on slow connections?
Video uses HLS with an adaptive-bitrate (ABR) quality ladder. The player automatically steps
down to a lower bitrate when bandwidth drops and steps back up when it recovers.

## What devices are supported?
Web (React portal), mobile on iOS and Android (originally React Native, later rewritten in
Flutter), and a TV app. The TV app supports signing in from your phone using device-grant
pairing instead of typing a password on the TV remote.

## Can I watch content in different languages?
Yes. The media pipeline supports multi-language audio dubs and subtitles. Subtitles are
converted from SRT to VTT using proxy-master injection, which avoids re-transcoding the video.

## What social features are available?
Short-form videos called Clips, threaded comments (two levels deep), @mentions, following
other users, and in-app calling powered by WebRTC with push notifications via FCM.

## How do payments and subscriptions work?
Subscriptions are handled through PayU and Razorpay. There is also a referral-earnings and
downline engine that rewards users for bringing in others.

## Is there any analytics?
Yes. A unified analytics dashboard tracks engagement, retention, content performance, and
active users.
