unit testmultiarray;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, multiarray, numerik;

type

  TTestMultiArray = class(TTestCase)
  published
    procedure TestBroadcastMatrixScalar;
    procedure TestBroadcastReshaped;
    procedure TestSlicing2D;
    procedure TestSlicing2DAddScalar;
    procedure TestSlicing2DAddTransposed;
    procedure TestTranspose;

    procedure TestReduceSimple;
    procedure TestReduceStacked;
    procedure TestReduceBig;
  end;

var
  M: TMultiArray;

implementation

procedure TTestMultiArray.TestBroadcastMatrixScalar;
begin
  M := TMultiArray([1, 2, 3, 4]).Reshape([2, 2]);
  AssertTrue(VectorsEqual((M + 2).GetVirtualData, [3, 4, 5, 6]));
end;

procedure TTestMultiArray.TestBroadcastReshaped;
begin
  M := [1, 2, 3];
  M := M.Reshape([3, 1]) * M.Reshape([1, 3]);
  AssertTrue(VectorsEqual(M.GetVirtualData, [1, 2, 3, 2, 4, 6, 3, 6, 9]));
end;

procedure TTestMultiArray.TestSlicing2D;
var
  M: TMultiArray;
begin
  M := TMultiArray(TSingleVector(Range(1, 10))).Reshape([3, 3]);
  M := M.Slice([[0, 2], [1]]); // ==> [[2], [8]]
  Assert(VectorsEqual(M.GetVirtualData, [2, 8]));
end;

procedure TTestMultiArray.TestSlicing2DAddScalar;
begin
  M := TMultiArray(TSingleVector(Range(1, 10))).Reshape([3, 3]);
  M := M.Slice([[0, 2], [1]]) + 10;
  Assert(VectorsEqual(M.GetVirtualData, [12, 18]));
end;

procedure TTestMultiArray.TestSlicing2DAddTransposed;
begin
  M := TMultiArray(TSingleVector(Range(1, 10))).Reshape([3, 3]);
  M := M.Slice([[0, 2], [1]]);
  M := M.T + M;
  AssertTrue(VectorsEqual(M.GetVirtualData, [4, 10, 10, 16]));
end;

procedure TTestMultiArray.TestTranspose;
begin
  M := [1, 2, 3, 4, 5, 6];
  M := M.Reshape([2, 3]);
  AssertTrue(VectorsEqual(M.T.GetVirtualData, [1, 4, 2, 5, 3, 6]));
end;

procedure TTestMultiArray.TestReduceSimple;
begin
  M := TMultiArray([1, 2, 3, 4, 5, 6]).Reshape([2, 3]);
  M := ReduceSum(M, 0);
  AssertTrue(VectorsEqual(M.GetVirtualData, [5, 7, 9]));
end;

procedure TTestMultiArray.TestReduceStacked;
begin
  M := TMultiArray([1, 2, 3, 4, 5, 6]).Reshape([2, 3]);
  M := ReduceSum(ReduceSum(M, 0), 0);
  AssertTrue(VectorsEqual(M.GetVirtualData, [21]));
end;

procedure TTestMultiArray.TestReduceBig;
var
  i: integer;
  expected: TSingleVector;
begin
  M := Random([10000, 100]);
  M := M / M;
  M := ReduceSum(M, 0);
  SetLength(expected, 100);
  for i := 0 to 99 do
    expected[i] := 10000;
  AssertTrue(VectorsEqual(M.GetVirtualData, expected));
end;

initialization
  RegisterTest(TTestMultiArray);

end.


