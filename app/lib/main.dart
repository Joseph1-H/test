import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

void main() => runApp(const JJApp());

class JJApp extends StatelessWidget {
  const JJApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'JJ AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: Colors.indigo,
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        colorSchemeSeed: Colors.indigo,
        brightness: Brightness.dark,
        useMaterial3: true,
      ),
      home: const ChatScreen(),
    );
  }
}

class ChatMessage {
  final String role; // 'user' or 'assistant'
  final String content;
  ChatMessage(this.role, this.content);
  Map<String, String> toJson() => {'role': role, 'content': content};
}

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  static const _defaultServer = 'http://raspberrypi.local:8000';

  final _messages = <ChatMessage>[];
  final _input = TextEditingController();
  final _scroll = ScrollController();
  String _serverUrl = _defaultServer;
  String _apiKey = '';
  bool _waiting = false;

  @override
  void initState() {
    super.initState();
    _loadSettings();
  }

  Future<void> _loadSettings() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _serverUrl = prefs.getString('serverUrl') ?? _defaultServer;
      _apiKey = prefs.getString('apiKey') ?? '';
    });
  }

  Future<void> _send() async {
    final text = _input.text.trim();
    if (text.isEmpty || _waiting) return;
    _input.clear();
    setState(() {
      _messages.add(ChatMessage('user', text));
      _waiting = true;
    });
    _scrollToEnd();

    try {
      final res = await http
          .post(
            Uri.parse('$_serverUrl/chat'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'messages': _messages.map((m) => m.toJson()).toList(),
              if (_apiKey.isNotEmpty) 'api_key': _apiKey,
            }),
          )
          .timeout(const Duration(minutes: 3));

      if (res.statusCode == 200) {
        final reply = jsonDecode(utf8.decode(res.bodyBytes))['reply'] as String;
        setState(() => _messages.add(ChatMessage('assistant', reply)));
      } else {
        _showError('Server error ${res.statusCode}: ${res.body}');
      }
    } catch (e) {
      _showError('Could not reach $_serverUrl\n$e');
    } finally {
      setState(() => _waiting = false);
      _scrollToEnd();
    }
  }

  void _showError(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(msg), duration: const Duration(seconds: 5)));
  }

  void _scrollToEnd() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients) {
        _scroll.animateTo(_scroll.position.maxScrollExtent,
            duration: const Duration(milliseconds: 250), curve: Curves.easeOut);
      }
    });
  }

  Future<void> _openSettings() async {
    final urlCtrl = TextEditingController(text: _serverUrl);
    final keyCtrl = TextEditingController(text: _apiKey);
    final saved = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Server settings'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: urlCtrl,
              decoration: const InputDecoration(labelText: 'Server URL'),
              keyboardType: TextInputType.url,
            ),
            TextField(
              controller: keyCtrl,
              decoration: const InputDecoration(labelText: 'API key (optional)'),
              obscureText: true,
            ),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Save')),
        ],
      ),
    );
    if (saved == true) {
      final prefs = await SharedPreferences.getInstance();
      final url = urlCtrl.text.trim().replaceAll(RegExp(r'/+$'), '');
      await prefs.setString('serverUrl', url);
      await prefs.setString('apiKey', keyCtrl.text.trim());
      setState(() {
        _serverUrl = url;
        _apiKey = keyCtrl.text.trim();
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Scaffold(
      appBar: AppBar(
        title: const Text('JJ AI'),
        actions: [
          IconButton(
            tooltip: 'New chat',
            icon: const Icon(Icons.add_comment_outlined),
            onPressed: () => setState(_messages.clear),
          ),
          IconButton(
            tooltip: 'Settings',
            icon: const Icon(Icons.settings_outlined),
            onPressed: _openSettings,
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: _messages.isEmpty
                  ? Center(
                      child: Text('Ask JJ anything',
                          style: TextStyle(color: scheme.onSurfaceVariant)),
                    )
                  : ListView.builder(
                      controller: _scroll,
                      padding: const EdgeInsets.all(12),
                      itemCount: _messages.length + (_waiting ? 1 : 0),
                      itemBuilder: (_, i) {
                        if (i == _messages.length) {
                          return const Align(
                            alignment: Alignment.centerLeft,
                            child: Padding(
                              padding: EdgeInsets.all(12),
                              child: SizedBox(
                                  width: 20,
                                  height: 20,
                                  child: CircularProgressIndicator(strokeWidth: 2)),
                            ),
                          );
                        }
                        final m = _messages[i];
                        final isUser = m.role == 'user';
                        return Align(
                          alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
                          child: Container(
                            margin: const EdgeInsets.symmetric(vertical: 4),
                            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                            constraints: BoxConstraints(
                                maxWidth: MediaQuery.of(context).size.width * 0.8),
                            decoration: BoxDecoration(
                              color: isUser ? scheme.primary : scheme.surfaceContainerHighest,
                              borderRadius: BorderRadius.circular(16),
                            ),
                            child: SelectableText(
                              m.content,
                              style: TextStyle(
                                  color: isUser ? scheme.onPrimary : scheme.onSurface),
                            ),
                          ),
                        );
                      },
                    ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 4, 12, 12),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _input,
                      minLines: 1,
                      maxLines: 5,
                      textInputAction: TextInputAction.send,
                      onSubmitted: (_) => _send(),
                      decoration: InputDecoration(
                        hintText: 'Message JJ…',
                        border: OutlineInputBorder(borderRadius: BorderRadius.circular(24)),
                        contentPadding:
                            const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  IconButton.filled(
                    onPressed: _waiting ? null : _send,
                    icon: const Icon(Icons.arrow_upward),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
