import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;
import 'package:url_launcher/url_launcher.dart';
import 'package:fl_chart/fl_chart.dart';
import 'dart:convert';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '反应谱在线计算',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const ResponseSpectrumPage(),
    );
  }
}

class ResponseSpectrumPage extends StatefulWidget {
  const ResponseSpectrumPage({super.key});

  @override
  State<ResponseSpectrumPage> createState() => _ResponseSpectrumPageState();
}

class _ResponseSpectrumPageState extends State<ResponseSpectrumPage> {
  final dampController = TextEditingController(text: '0.05');
  final dtController = TextEditingController(text: '0.02');
  final tpController = TextEditingController(text: '6.0');
  final dtpController = TextEditingController(text: '0.02');

  PlatformFile? selectedFile;
  bool loading = false;
  String? message;
  String? downloadUrl;
  dynamic result;

  final String baseUrl = 'http://127.0.0.1:8000';

  Future<void> pickFile() async {
    final picked = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['txt', 'csv'],
      withData: true,
    );

    if (picked != null && picked.files.isNotEmpty) {
      setState(() {
        selectedFile = picked.files.first;
        message = null;
      });
    }
  }

  Future<void> uploadAndCalculate() async {
    if (selectedFile == null || selectedFile!.bytes == null) {
      setState(() {
        message = '请先选择 txt 或 csv 文件';
      });
      return;
    }

    setState(() {
      loading = true;
      message = null;
      result = null;
      downloadUrl = null;
    });

    try {
      final uri = Uri.parse('$baseUrl/api/response-spectrum');
      final request = http.MultipartRequest('POST', uri);

      request.fields['damp'] = dampController.text;
      request.fields['dt'] = dtController.text;
      request.fields['Tp'] = tpController.text;
      request.fields['dtp'] = dtpController.text;

      request.files.add(
        http.MultipartFile.fromBytes(
          'file',
          selectedFile!.bytes!,
          filename: selectedFile!.name,
        ),
      );

      final response = await request.send();
      final body = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        final data = jsonDecode(body);
        debugPrint(data['result'].toString());

        setState(() {
          result = data['result'];
          downloadUrl = '$baseUrl${data['download_url']}';
          message = '计算成功';
        });
      } else {
        final data = jsonDecode(body);
        setState(() {
          message = '计算失败：${data['detail']}';
        });
      }
    } catch (e) {
      setState(() {
        message = '请求失败：$e';
      });
    } finally {
      setState(() {
        loading = false;
      });
    }
  }

  Future<void> downloadResult() async {
    if (downloadUrl == null) return;

    final uri = Uri.parse(downloadUrl!);
    await launchUrl(uri, mode: LaunchMode.externalApplication);
  }

  Widget buildInput(String label, TextEditingController controller) {
    return TextField(
      controller: controller,
      keyboardType: TextInputType.number,
      decoration: InputDecoration(
        labelText: label,
        border: const OutlineInputBorder(),
      ),
    );
  }

  List<double> getList(String key) {
    if (result == null || result[key] == null) return [];
    return List<double>.from(
      result[key].map((e) => (e as num).toDouble()),
    );
  }

  @override
  Widget build(BuildContext context) {
    final tData = getList('T');
    final arsData = getList('ARS');
    final vrsData = getList('VRS');
    final drsData = getList('DRS');
    final pvrsData = getList('PVRS');
    final parsData = getList('PARS');

    return Scaffold(
      appBar: AppBar(
        title: const Text('反应谱在线计算'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: ListView(
          children: [
            ElevatedButton.icon(
              onPressed: pickFile,
              icon: const Icon(Icons.upload_file),
              label: const Text('选择地震波文件 txt/csv'),
            ),
            const SizedBox(height: 8),
            Text(selectedFile == null ? '未选择文件' : '已选择：${selectedFile!.name}'),

            const SizedBox(height: 20),

            Row(
              children: [
                Expanded(
                  child: buildInput('阻尼比', dampController),
                ),
                const SizedBox(width: 12),

                Expanded(
                  child: buildInput('时间步长', dtController),
                ),
                const SizedBox(width: 12),

                Expanded(
                  child: buildInput('最大周期', tpController),
                ),
                const SizedBox(width: 12),

                Expanded(
                  child: buildInput('周期间隔', dtpController),
                ),
              ],
            ),

            const SizedBox(height: 20),

            ElevatedButton(
              onPressed: loading ? null : uploadAndCalculate,
              child: loading
                  ? const SizedBox(
                      width: 22,
                      height: 22,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text('开始计算'),
            ),

            if (message != null) ...[
              const SizedBox(height: 16),
              Text(message!),
            ],

            if (downloadUrl != null) ...[
              const SizedBox(height: 16),
              ElevatedButton.icon(
                onPressed: downloadResult,
                icon: const Icon(Icons.download),
                label: const Text('下载 CSV 结果'),
              ),
            ],

             if (result != null) ...[
              const SizedBox(height: 24),

              SpectrumChart(
                title: '加速度反应谱 ARS',
                xLabel: 'T / s',
                yLabel: 'ARS',
                xData: tData,
                yData: arsData,
              ),

              const SizedBox(height: 20),

              SpectrumChart(
                title: '速度反应谱 VRS',
                xLabel: 'T / s',
                yLabel: 'VRS',
                xData: tData,
                yData: vrsData,
              ),

              const SizedBox(height: 20),

              SpectrumChart(
                title: '位移反应谱 DRS',
                xLabel: 'T / s',
                yLabel: 'DRS',
                xData: tData,
                yData: drsData,
              ),

              const SizedBox(height: 20),

              SpectrumChart(
                title: '伪速度反应谱 PVRS',
                xLabel: 'T / s',
                yLabel: 'PVRS',
                xData: tData,
                yData: pvrsData,
              ),

              const SizedBox(height: 20),

              SpectrumChart(
                title: '伪加速度反应谱 PARS',
                xLabel: 'T / s',
                yLabel: 'PARS',
                xData: tData,
                yData: parsData,
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class SpectrumChart extends StatelessWidget {
  final String title;
  final String xLabel;
  final String yLabel;
  final List<double> xData;
  final List<double> yData;

  const SpectrumChart({
    super.key,
    required this.title,
    required this.xLabel,
    required this.yLabel,
    required this.xData,
    required this.yData,
  });

  @override
  Widget build(BuildContext context) {
    final spots = <FlSpot>[];

    for (int i = 0; i < xData.length && i < yData.length; i++) {
      spots.add(FlSpot(xData[i], yData[i]));
    }

    final maxX = xData.isEmpty ? 1.0 : xData.reduce((a, b) => a > b ? a : b);
    final maxY = yData.isEmpty ? 1.0 : yData.reduce((a, b) => a > b ? a : b);

    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: SizedBox(
          height: 330,
          child: Column(
            children: [
              Text(
                title,
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 6),
              Text('$xLabel    $yLabel'),
              const SizedBox(height: 12),
              Expanded(
                child: LineChart(
                  LineChartData(
                    minX: 0,
                    maxX: maxX,
                    minY: 0,
                    maxY: maxY == 0 ? 1 : maxY * 1.1,
                    gridData: const FlGridData(show: true),
                    titlesData: const FlTitlesData(show: true),
                    borderData: FlBorderData(show: true),
                    lineBarsData: [
                      LineChartBarData(
                        spots: spots,
                        isCurved: false,
                        dotData: const FlDotData(show: false),
                        barWidth: 2,
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}