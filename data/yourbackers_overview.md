# YourBackers — Platform Overview

YourBackers is an OTT (over-the-top) streaming and social platform built by Gravitorix.
It combines on-demand video streaming with a companion social network, delivered across
web, mobile (iOS/Android), and TV clients.

## Streaming

- Video is delivered using HLS (HTTP Live Streaming) with an adaptive-bitrate (ABR)
  quality ladder, so playback quality adjusts automatically to the viewer's bandwidth.
- Premium content is protected with Multi-DRM using Pallycon Widevine. DRM playback was
  verified end-to-end on web and Android.
- Media files are stored in a private Amazon S3 bucket and served through a CloudFront CDN
  using signed URLs, so only authorized viewers can fetch the content.
- A media pipeline supports multi-language audio dubs and converts SRT subtitles to VTT via
  proxy-master injection, avoiding costly re-transcoding.
- Content is organized as movies and series, with watch progress tracked at the
  season and episode level.

## Social platform

- Short-form video feature called "Clips".
- Two-level threaded comments and an app-wide @mention system.
- Follow relationships between users.
- In-app calling built on WebRTC, with FCM push notifications used to wake the app for
  incoming calls.

## Clients

- Cross-platform mobile app built in React Native, later rewritten in Flutter.
- A TV application that supports "sign-in-on-your-phone" device-grant pairing.
- A React-based web client portal.

## Monetization

- Subscription payments integrated with PayU and Razorpay.
- A referral-earnings and downline engine.
- A unified analytics dashboard tracking engagement, retention, content performance,
  and active users.
